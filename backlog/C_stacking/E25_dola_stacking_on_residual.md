# E25 — DoLa Stacks on Any Residual Steer for Additive Factuality Gain

> **One-line claim:** DoLa (decoding-time) stacks on any residual steer for
> additive factuality gain at no coherence cost.
>
> **Source design space:** Block C — Stacking and Multi-Vector Composition (E17–E26).
>
> **Implementation status:** `PENDING — UNTESTED`. Requires DoLa decoding
> implementation and TruthfulQA evaluation setup.

---

## In Plain English

**What we're testing, simply:** "DoLa" is a separate trick that makes a model more
truthful — it works at the very last moment, when the model is choosing each word,
not by editing the model's "thoughts." Because it acts somewhere different, the
hope is it can be added *on top of* any of our regular nudges to gain extra
truthfulness for free, without harming the text.

**Key terms (defined here):**
- **Steering / steering vector:** nudging the model by adding a direction to its
  internal "thoughts"; the direction is the steering vector ("residual steer").
- **Residual stream:** the model's running internal "thoughts" that a normal nudge
  edits.
- **Layer:** the model's stacked processing steps. **DoLa** works by comparing
  early layers against late layers to favor more confident, factual words.
- **Alpha / strength:** how hard a nudge pushes.
- **DiffMean:** the simple recipe for building a nudge.
- **Coherence:** whether the text stays fluent and sensible.
- **Stacking:** using two methods together. Here one is a normal nudge to the
  thoughts, and the other (DoLa) acts at word-selection time — *different places*,
  so they're expected to add up rather than clash.
- **Factuality:** how often the answers are actually true (measured on a
  truthfulness quiz).

**Why we're doing this (the point):** If a truthfulness booster simply layers onto
whatever else we're steering, with no downside, that's a free win for any setup.

**What the result would mean:** A win means we can bolt truthfulness onto any
steering recipe at no cost to readability. A loss means DoLa trades against the
other nudge or hurts the text, so it isn't free after all.

See [`../GLOSSARY.md`](../GLOSSARY.md) for any other term.

---

## 1. Motivation (>= 100 words)

DoLa (Decoding by Contrasting Layers, arXiv:2309.03883) is a decoding-time
factuality enhancement that requires no steering vector and no activation edit.
It works by contrasting the output logit distributions of an early layer and
a late layer of the same model: the logit difference amplifies predictions
that the late layer has "updated" over the early layer, which tends to be
the factual content. Because DoLa acts at the decoding / logit level (AXIS 1:
site = logits), it operates entirely after all residual-stream steering has
completed. The intervention sites are disjoint: residual CAA edits h during
the forward pass; DoLa reads the logits after the entire forward pass and
applies a contrast before sampling. This makes DoLa the ideal "last layer"
in the steering stack: it can be combined with any residual-stream method
without interaction. The key claim is that the factuality gain from DoLa
(measured on TruthfulQA) is additive over any residual steer's factuality
gain, and the combined method does not degrade coherence (PPL stays flat).
This would make DoLa a universally composable factuality booster — an always-
add component in any steering stack.

---

## 2. Formal Hypothesis (>= 50 words)

Because DoLa's intervention site (logit layer) is strictly after and strictly
disjoint from the residual stream's intervention sites (layers 1 through L-1),
and because DoLa's operation (logit contrastive decoding) does not depend on
the content of h at any layer — it only reads the final output distribution
— the DoLa factuality gain should not be reduced by any preceding residual
steering edit. In fact, if residual steering has improved the behavioral
content of the generation (e.g., safety refusal, politeness), DoLa should
further improve the factuality of the remaining content without affecting the
behavioral signal. Formal claim: TruthfulQA MC accuracy with [residual steer
+ DoLa] equals or exceeds max(accuracy with residual steer alone, accuracy
with DoLa alone), i.e., the gain is at least additive, at matched PPL (within
0.3 logPPL of residual-steer-alone baseline), at 3-seed median on Gemma-2-2B-it.

---

## 3. Falsifier (>= 30 words)

If TruthfulQA accuracy with [residual steer + DoLa] is lower than DoLa alone
by more than 2 percentage points (at 3-seed median), the residual steer is
disrupting DoLa's contrastive signal, and the two methods are NOT fully
composable at the logit level. If PPL with [residual steer + DoLa] exceeds
residual-steer-alone by more than 0.3 logPPL, DoLa is adding coherence cost
in the combined setting (unexpected) and the null-cost claim is FALSIFIED.

---

## 4. Citations (Citation Rigor >= 80 words)

```
Chuang, Yung-Sung, et al. 2023 arXiv 'DoLa: Decoding by Contrasting Layers
Improves Factuality in Large Language Models' (arXiv:2309.03883) — the primary
method under test; introduces logit-level contrastive decoding between early
and late layers; reports factuality gains on TruthfulQA and other benchmarks
[NEEDS VERIFICATION on Gemma-2-2B]; the disjoint-site claim is the mechanism
for why DoLa should compose with any residual steer.

[Steering-stackable-vs-competing-analysis.md §2.3, this project]: "Residual
addition × contrastive decoding (DoLa 2309.03883) — DoLa acts at the logits
by contrasting layers; it sits AFTER all residual edits and composes with them."
— the design principle that licenses this experiment.

[N5 geometry result, C2, this project, SUPPORTED]: logPPL = 5.40 + 2.87 *
offshell, R² = 0.81 — DoLa does not increase offshell displacement (it is a
logit operation, not a residual edit); therefore N5 predicts no additional PPL
cost from adding DoLa to a residual steer.

Arditi, Andy, et al. 2024 arXiv 'Refusal in Language Models Is Mediated by a
Single Direction' (arXiv:2406.11717) — the refusal direction is the residual
steer being combined with DoLa; if DoLa disrupts refusal efficacy (via logit
reweighting), the composition is not fully additive.
```

---

## 5. Mechanism

### 5.1 DoLa decoding

At each generation step t:

    logits_late = model.late_layer_logits(h_T)   (final layer output)
    logits_early = model.early_layer_logits(h_k)  (chosen early layer)
    logits_DoLa = logits_late - lambda * logits_early
    token_t = sample(softmax(logits_DoLa))

The lambda weight controls the strength of the contrastive signal.

### 5.2 Interaction with residual steering

The residual steering modifies h_T before it reaches the final layer (the
forward pass includes the residual edit at layer L_b < L). Therefore:

    logits_late^steered = final_layer_projection(h_T + alpha*v_behavior_contribution)

The DoLa contrast is then:

    logits_DoLa^steered = logits_late^steered - lambda * logits_early^steered

The steered h_T shifts the late-layer logits; the early-layer logits are shifted
by the residual edit too (the edit propagates through all subsequent layers).
The contrastive cancellation removes the "early layer" component — which is
mostly the token-level and syntactic information. The residual edit's behavioral
signal, accumulated through layers L_b to L, should survive the contrast.

### 5.3 Expected factuality composition

If the residual steer for refusal reduces factuality (the model refuses more
but states fewer facts overall), and DoLa improves factuality of the remaining
non-refused output, the combined method should have: (a) refusal behavior
from the residual steer; (b) improved factuality in non-refused responses
from DoLa. The TruthfulQA measurement on non-refused responses is the key metric.

---

## 6. Predicted Delta

| Metric | Predicted value | Rationale |
|---|---|---|
| TruthfulQA (DoLa alone) | [50%, 65%] | DoLa standard range [NEEDS VERIFICATION] |
| TruthfulQA (residual steer alone) | [45%, 60%] | Refusal steer may reduce factuality |
| TruthfulQA (residual + DoLa) | >= max(above) + 3 pp | Additive gain claim |
| PPL (residual + DoLa vs residual alone) | [−0.1, +0.3] | Near-zero DoLa PPL cost |
| Refusal behavior (residual + DoLa) | >= 95% of residual-alone | DoLa doesn't disrupt refusal |

Pre-registered; [NEEDS VERIFICATION] on Gemma-2-2B.

---

## 7. Experimental Protocol

### 7.1 Primary experiment

- **Model:** Gemma-2-2B-it, 4-bit
- **Conditions:** (1) no steer, (2) DoLa alone, (3) residual steer alone,
  (4) residual steer + DoLa
- **DoLa early layer:** layer 6 or 10 (to be determined by DoLa paper's
  recommendation for Gemma-class models [NEEDS VERIFICATION])
- **DoLa lambda:** 1.0 (default per paper)
- **Residual steer:** refusal vector at alpha_opt from E9
- **Metrics:** TruthfulQA MC accuracy (4 conditions); PPL; refusal efficacy
- **Seeds:** 3 (screening)
- **Wall-clock:** ~1.5 h on 4090 (DoLa adds minimal overhead per token)

### 7.2 Where it shines

DoLa's factuality gain is largest on questions requiring factual recall
(TruthfulQA) rather than behavioral compliance. The residual steer's behavioral
effect and DoLa's factuality effect should be largely orthogonal, making the
combined method particularly effective for "factually honest AND behaviorally
safe" generation.

---

## 8. Cross-References

- **E50** (minimal stack SOTA recipe): DoLa is one of the composable components;
  E25 validates its composability
- **E22** (norm budget): DoLa adds zero offshell displacement — it is not
  constrained by the N5 budget (pure logit operation)
- **IDEA_TABLE.md** Block C row E25
- **N5** (norm budget, SUPPORTED): DoLa is outside the N5 budget domain (logit,
  not residual) — predicted PPL cost from DoLa alone is zero by N5

---

## 9. Committee Q&A

**Q: Does DoLa's early-layer contrastive signal interact with the residual
steering's layer-specific effects?**

> The DoLa contrast (late-layer minus early-layer logits) cancels out features
> that are present in both early and late layers. The residual steering adds
> features to specific mid-late layers (L_b), which are not present in the
> early layer (L_c < L_b). Therefore, the steering-added features survive the
> DoLa contrast. This is the mechanism for why DoLa doesn't disrupt the
> behavioral signal from residual steering.

**Q: Does TruthfulQA measure factuality on refused outputs?**

> TruthfulQA MC (multiple-choice) format is less susceptible to refusal than
> open-ended TruthfulQA. The MC format will be used primarily; open-ended
> TruthfulQA will be reported separately with refused answers excluded.

---

## 10. Verification Checklist

- [ ] DoLa implementation validated on a test example (logit contrast verified)
- [ ] Early layer for DoLa on Gemma-2-2B determined (per paper recommendation)
- [ ] TruthfulQA MC accuracy measured for all 4 conditions (3 seeds each)
- [ ] PPL measured for all 4 conditions
- [ ] Refusal efficacy measured for conditions 3 and 4
- [ ] Additive gain claim verified: condition 4 >= max(condition 2, condition 3)
- [ ] IDEA_TABLE.md row E25 updated

---

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: UNTESTED / PENDING. N5 (SUPPORTED)
  predicts zero offshell from DoLa (logit-level, not residual). DoLa's disjoint-
  site property is mechanism-grounded. arXiv:2309.03883 is a verified paper.
  Code needed: DoLa decoding hook (replaces standard sampling with contrastive
  logit sampling); TruthfulQA MC evaluation. Blocked on residual steer
  infrastructure from E9.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-C.*

### Prior plausibility

HIGH. The disjoint-site argument is the strongest composability argument in
Block C (logit vs residual truly disjoint). DoLa has been demonstrated to
work on multiple models. The main uncertainty is whether Gemma-2-2B's layer
structure is appropriate for DoLa's contrastive approach (the paper recommends
different early-layer choices for different architectures).

### Mechanism scrutiny

The mechanism is sound. The key question is whether the residual steering
modifies the early-layer logits (through its downstream propagation) in a way
that reduces the DoLa contrast's factuality signal. The protocol should verify
that DoLa's TruthfulQA gain is maintained in the combined condition.

### Confounds

1. **Early-layer choice for DoLa:** the optimal early layer varies by model;
   a suboptimal choice may not show the full DoLa benefit.
2. **Behavioral vs factual content:** for refusal-steered outputs, many
   TruthfulQA questions may be refused entirely, reducing the number of
   scoreable responses. The protocol should report coverage (fraction of
   TruthfulQA questions answered) alongside accuracy.

### Verdict

**NOVEL+TESTABLE.** Low compute cost. The disjoint-site composability claim
is the cleanest in Block C and will validate a key principle of the 7-axis
decision rule.

---

## Pseudocode & Methodology

This section specializes [`../METHODOLOGY.md`](../METHODOLOGY.md) to stacking DoLa
(a logit-level operation) on a residual steer — disjoint sites that should compose
additively. Status: `UNTESTED` — needs a DoLa contrastive-decoding hook + TruthfulQA MC.

### 1. Steering-vector recipe (one residual DiffMean; DoLa adds NO vector)

```python
# Residual arm: closed-form DiffMean refusal vector (METHODOLOGY §1.3), injected at L_b.
v = normalize(extract.build_vector_bank(model,tok,load_concept("refusal"),L_b)[L_b]["diffmean"])
# DoLa arm: NO steering vector at all — a decoding-time logit contrast between layers (§5.1):
#   logits_DoLa = logits_late(h_T) - lambda * logits_early(h_k)
# Sites are disjoint: residual edit at layers 1..L-1; DoLa reads final logits AFTER the pass.
```

### 2. Experiment procedure

```text
conditions = { no_steer,
               dola_alone        : contrastive logit decode, no residual edit,
               residual_alone    : hooks.apply_operation(h, v, "add", alpha) at L_b,
               residual_plus_dola: residual edit THEN DoLa logit contrast }
for cond in conditions:
  for seed in 1..3:
    measure: TruthfulQA MC accuracy (eval.mcq_accuracy-style ground truth),
             PPL (eval.perplexity), refusal efficacy,
             geometry.offshell_displacement   # DoLa adds ZERO offshell (logit-only; N5 domain excludes it)
```

### 3. Measurement & decision rule

- PRIMARY metric: TruthfulQA MC accuracy of [residual + DoLa] vs `max(residual-alone, DoLa-alone)`.
- Pre-registered FALSIFIER (§3): [residual + DoLa] accuracy lower than DoLa-alone by
  `> 2 pp` ⇒ not composable; OR PPL exceeds residual-alone by `> 0.3` logPPL ⇒ null-cost
  claim FALSIFIED.
- Secondary (§6): refusal behavior `>= 95%` of residual-alone (DoLa doesn't disrupt it).

### 4. Where the code is / status

The residual `add` path exists. MISSING: a **DoLa decoding hook** (replace standard
sampling with the early/late logit contrast; arXiv:2309.03883 is a verified paper) and
**TruthfulQA MC** evaluation. By METHODOLOGY §3, DoLa contributes no residual
displacement, so the composite's geometry term is unaffected by the DoLa arm. `UNTESTED`.

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/E25.md`.
