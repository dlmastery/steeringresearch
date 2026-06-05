# E9 — CAST Harmless-Input Refusal Gate

> **One-line claim:** CAST thresholding keeps harmless-input refusal below 3%
> while raising harmful-input refusal by at least 50 percentage points versus
> unconditional steering at matched alpha.
>
> **Source design space:** Block B — Conditional / Gated Steering (E9–E16).
>
> **Implementation status:** `PENDING — UNTESTED`. Multi-vector CAST code and
> the dual-forward probe infrastructure required by the Rogue Scalpel guard have
> not yet been built. See Section 11 for what is needed.

---

## In Plain English

**What we're testing, simply:** A refusal nudge that's always on makes the model
refuse *everything*, even harmless questions — annoying and useless. This asks
whether a "gate" can switch the nudge on only when a request is actually harmful, so
normal requests are left untouched.

**Key terms (defined here so you don't have to look anything up):**
- **Language model (LLM):** an AI that predicts the next word; here, small Gemma
  models.
- **Steering:** nudging the model's behavior by adding a direction to its internal
  state while it writes.
- **Steering vector:** the specific direction of the nudge (here, a "refuse" nudge).
- **Residual stream / hidden state:** the model's running internal "thought" state,
  where the nudge is added and which the gate reads.
- **Layer:** one of the model's stacked processing steps.
- **DiffMean:** the simple recipe for building a direction from "yes/no" examples
  (here, harmful vs. harmless prompts).
- **alpha (strength):** how hard the nudge pushes.
- **Conditional steering:** only applying the nudge *when it's relevant*, instead of
  always.
- **The condition / gate:** a quick check that decides whether to fire the nudge. It
  measures how close the current request looks to "harmful" examples, and fires only
  if it crosses a set line.
- **CAST:** the name of this gating method (Conditional Activation Steering).
- **Threshold:** the cut-off line the gate uses to decide harmful vs. harmless.
- **Read-only probe:** the gate just *looks* at the internal state, it doesn't change
  it — so checking is essentially free and can't itself break the text.
- **Over-refusal / false refusal:** wrongly refusing a harmless request. We want this
  under 3%.
- **Safety:** whether the model refuses genuinely harmful requests (measured on a
  jailbreak test).

**Why we're doing this (the point):** Safety steering is only deployable if it
catches harmful requests without blocking everyday ones. This tests whether the gate
delivers that precision — strong refusal on harmful inputs, almost none on benign
ones.

**What the result would mean:** A positive result means we can add a safety nudge
that stays quiet on normal use — the foundation for every other gated experiment in
this block. A negative result means the gate can't separate harmful from harmless
cleanly, so conditional safety steering isn't practical yet.

See [`../GLOSSARY.md`](../GLOSSARY.md) for any other term.

---

## 1. Motivation (>= 100 words)

Activation steering — adding a fixed behavior vector alpha*v to the residual
stream at inference time — is among the most computationally cheap ways to
modulate model behavior. The price is bluntness: an always-on refusal vector
fires on every input, including the benign majority, producing over-refusal
that erodes utility and user trust. The Rogue Scalpel finding (arXiv:2509.22067)
makes this worse: even low-alpha unconditional steers raise harmful compliance
on a fraction of inputs by displacing the fragile mid-layer refusal ridge, so
the "safety tax" is not just over-refusal but also mis-refusal. CAST (Conditional
Activation Steering, arXiv:2409.05907) addresses bluntness by adding a condition
check: compute a condition vector from contrastive prompt activations (harmful vs
harmless examples), and fire the behavior vector only when the cosine similarity
of the current hidden state to the condition vector exceeds a threshold theta_c.
The gate is a read-only probe — it consumes no norm budget (corpus:
steering-stackable-vs-competing-analysis.md §1) and therefore does not
constitute a competing intervention with the downstream behavior write. This
experiment tests whether CAST's gating mechanism achieves the precision that
makes safety steering practically deployable: a near-zero false-refusal rate
on benign inputs paired with a substantial true-refusal uplift on harmful ones.

---

## 2. Formal Hypothesis (>= 50 words)

Because CAST's condition vector is derived from the mean difference of activations
on harmful versus harmless prompts (the same DiffMean mechanism as behavior
vectors, per E4's cosine-alignment result), it encodes the semantic contrast
between safety-relevant and safety-irrelevant inputs in the same linear subspace
that governs refusal. When this condition vector is used as a gate — fire behavior
vector only if sim(h_early, v_condition) > theta_c — the gate is orthogonal to
the behavior write on AXIS 5 (condition) without touching AXIS 3 (coefficient)
or AXIS 2 (direction), preserving all of the behavior vector's properties on
harmful inputs while leaving benign inputs unsteered. Formal claim: on Gemma-2-2B-it
evaluated against a 200-prompt balanced set (100 harmful from JailbreakBench,
100 benign from AlpacaEval), CAST gating at the optimal theta_c reduces harmless-
input refusal from the unconditional rate (expected > 20% at alpha producing
>= 50% harmful refusal) to below 3%, while maintaining harmful-input refusal
at least 50 percentage points above no-steering baseline.

---

## 3. Falsifier (>= 30 words)

If, after theta_c grid search over {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7},
no threshold simultaneously achieves harmless-refusal < 3% AND harmful-refusal
delta >= 50 pp above the no-steering baseline at 3-seed median, this hypothesis
is DISCARDED and Status moves to FALSIFIED. Additionally, if the optimal CAST
operating point does not Pareto-dominate unconditional steering on the joint
(harmful-refusal, harmless-refusal) plane — i.e., if CAST achieves the 3% target
only by also reducing harmful refusal below baseline — the gating mechanism
provides no benefit and the hypothesis is also DISCARDED.

---

## 4. Citations (Citation Rigor >= 80 words)

```
Wu, Yuming, et al. 2024 arXiv 'Conditional Activation Steering: Concept-Level
Control via Conditional Vectors' (arXiv:2409.05907) — the primary method
this experiment tests; introduces condition vectors from contrastive PCA and
cosine-threshold gating; reports harmless-refusal reduction on GPT-style models
[NEEDS VERIFICATION on Gemma-2-2B]; the gating mechanism is the direct object
of this hypothesis.

Korznikov, A., et al. 2026 ICML 'The Rogue Scalpel: Activation Steering
Compromises LLM Safety' (arXiv:2509.22067) — establishes that unconditional
steering (including benign vectors) raises harmful compliance via off-manifold
mid-layer displacement; provides the threat model that motivates conditional
gating as Guard Layer E; their JailbreakBench protocol is inherited here.

Arditi, Andy, et al. 2024 arXiv 'Refusal in Language Models Is Mediated by a
Single Direction' (arXiv:2406.11717) — characterizes the refusal direction as
a linear subspace; motivates why a condition vector (derived from the same
contrastive mechanism) should be able to gate on harm-relevance without touching
the refusal direction itself (N6 / E32 implication).

Tang, Haoran, et al. 2025 arXiv 'FineSteer: Fine-Grained Steering of Large
Language Models via Subspace-Guided Conditional Activation Steering'
(arXiv:2604.15488) — direct CAST successor using energy-ratio gating (SCS);
provides a comparative gate PR-AUC baseline that E12 will exploit; relevant
here as the strongest known competitor to the cosine-threshold CAST gate.
```

---

## 5. Mechanism

### 5.1 CAST gating in the 7-axis framework

CAST is a setting on AXIS 5 (WHEN / condition):

- AXIS 1 (site): early-layer read (condition), mid-late-layer write (behavior)
- AXIS 2 (direction): v_condition (condition), v_behavior (refusal)
- AXIS 5 (condition): gate fires iff cos(h_layer_k, v_condition) > theta_c
- All other axes: identical to unconditional CAA

The condition check is a dot product on the hidden state — a read with zero
norm impact. The gate therefore adds no off-shell displacement (the norm-budget
law N5: log PPL = 5.40 + 2.87 * offshell, R² = 0.81, C2) — the condition
probe does not consume the norm budget that the behavior write will spend.

### 5.2 Protocol sketch

```python
def cast_gate(h_early, v_condition, theta_c):
    """Returns True iff the input is condition-relevant."""
    sim = F.cosine_similarity(h_early, v_condition.unsqueeze(0), dim=-1)
    return (sim > theta_c).float()   # shape: (batch,)

def cast_steer(h_late, v_behavior, alpha, gate_mask):
    """Apply behavior vector only where gate fired."""
    return h_late + alpha * gate_mask.unsqueeze(-1) * v_behavior
```

Layer choices: condition read at layer L_c in [6, 12] (early-middle, pre-refusal
ridge); behavior write at layer L_b in [14, 20] (post-condition, CAST canonical
site for Gemma-2-2B); L_c < L_b is the required ordering (E26 tests the reverse).

### 5.3 Expected operating point

From E3 (SUPPORTED): the coherence cliff for a refusal vector on Gemma-2-2B
occurs at alpha ~ 15-20 (relative), above which PPL rises super-linearly.
Unconditional steering at alpha in the safe zone still causes > 20% harmless
refusal (corpus estimate). CAST should reduce this by restricting firing to
inputs whose early-layer representation resembles the harmful training examples,
recovering selectivity without sacrificing harmful refusal.

---

## 6. Predicted Delta

| Metric | Predicted value | Rationale |
|---|---|---|
| Harmless refusal rate (CAST) | < 3% | Core claim; CAST selectivity target |
| Harmful refusal rate (CAST) | >= 50 pp above no-steer | Core claim |
| Harmful refusal rate (unconditional at same alpha) | 60-80% | E3 supported range |
| Harmless refusal rate (unconditional) | 20-40% | E3 supported; over-refusal known |
| MMLU delta (CAST vs no-steer, benign set) | [0, +1.5] pp | Gate fires rarely on MMLU; expected near-zero tax |
| PPL on harmless prompts (CAST vs no-steer) | [0, +0.3] | Gate suppresses write; PPL should be clean |
| Composite improvement vs unconditional | [+0.05, +0.15] | Selectivity and capability axes dominate |
| Gate AUC (theta_c = optimal) | [0.80, 0.92] | CAST paper reports ~0.85 [NEEDS VERIFICATION] |

All values are pre-registered; [NEEDS VERIFICATION] until reproduced on 4090.

---

## 7. Experimental Protocol

### 7.1 Primary experiment

- **Model:** Gemma-2-2B-it, 4-bit (bnb), smoke on Gemma-3-1B-it first
- **Contrast pairs:** 50 harmful / 50 harmless prompts (E1 result: 50 pairs
  reaches >= 90% asymptotic effect; SUPPORTED)
- **Condition vector:** DiffMean of early-layer (L_c) activations on harmful vs
  harmless prompts; DiffMean chosen over PCA (E4 SUPPORTED: cosine = 0.994-0.996)
- **Behavior vector:** refusal direction (DiffMean, late layer L_b)
- **Theta_c grid:** {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7} — 7 points
- **Alpha grid:** {5, 10, 15, 20} (relative, E7 norm convention)
- **Evaluation set:** 100 harmful (JailbreakBench subset) + 100 harmless
  (AlpacaEval) — balanced, fixed seed
- **Metrics:** harmful-refusal rate, harmless-refusal rate, MMLU on benign set,
  PPL on harmless set, gate AUC (PR-AUC at optimal theta_c)
- **Seeds:** 3 for screening; 7 if screening passes (Section 7 / CLAUDE.md)
- **Wall-clock estimate:** ~ 2-3 hours on 4090 for full grid (7 * 4 * 3 seeds)

### 7.2 Where it shines

This experiment shines when harmful and harmless inputs are maximally separable
in early-layer representation space (high Fisher ratio at L_c). If E2 had
shown Spearman >= 0.7 (it was FALSIFIED at rho=0.14 on Gemma-270m), that would
undermine confidence; on Gemma-2-2B at the correct layer the separability is
expected to recover. The ideal demo prompt set is safety-topic prompts (hate
speech, illegal advice) vs factual question prompts — maximal contrast.

### 7.3 Dependencies

- E1 (pair count): SUPPORTED — 50 pairs is sufficient
- E4 (DiffMean vs PCA): SUPPORTED — DiffMean is valid for condition vector
- E7 (norm-relative alpha): SUPPORTED — use relative alpha convention
- N5 (norm budget): SUPPORTED — condition read does not consume budget

---

## 8. Cross-References

- **IDEA_TABLE.md** Block B row E9 (Status = PENDING)
- **E10** (condition vector orthogonality): prerequisite for OR-gating E11
- **E11** (OR-gate coverage): multi-category extension of this experiment
- **E12** (energy-ratio vs cosine gate): ablation comparing CAST cosine to SCS
- **E13** (early condition layer): layer-sensitivity of the condition read
- **E14** (discriminative-layer): alternative gate architecture
- **E15** (learned gate): learned logistic replacing fixed theta_c
- **E16** (capability tax): capability impact of gated vs always-on
- **N5** (norm budget): condition gate adds zero budget cost — free safety guard
- **N6** (gate in read not write): the architectural principle CAST instantiates
- **Rogue Scalpel guard** (corpus/steering-first-principles-v2): Guard Layer E
  is exactly this experiment's gating mechanism

---

## 9. Committee Q&A

**Q: Does CAST gating actually beat a prompt classifier for safety routing?**

> A prompt classifier (e.g., a DistilBERT toxicity head) also gates on input.
> CAST's advantage is: (a) it reads activations of the steered model, so the
> gate signal is drawn from the same representational space as the behavior
> vector — there is no domain shift between gating and acting; (b) it adds no
> latency beyond a dot product (no second-model forward pass); (c) it composes
> natively with the 7-axis framework. Whether it beats a small BERT classifier
> in AUC is an empirical question; E15 tests learned gates and could confirm
> or challenge this. The falsifier (Section 3) is neutral on the comparison
> with classifiers — we claim only that CAST beats unconditional steering.

**Q: If the condition vector is also derived by DiffMean, isn't it just a
second behavior vector?**

> Functionally it is the same extraction mechanism but acts on AXIS 5 (condition),
> not AXIS 4 (operation). The critical distinction: the condition vector is read
> at an early layer and no write occurs; the behavior vector is written at a
> later layer only when the gate fires. The AXIS separation is mechanical, not
> just conceptual — L_c < L_b, and the condition probe touches no gradient path
> during inference.

**Q: What if the harmful / harmless split is not linearly separable at L_c?**

> Then the gate AUC will be < 0.70 and the hypothesis will not reach its
> harmless-refusal target (the gate will fire on benign inputs). This is a
> real risk on Gemma-3-1B-it, where E2 showed rho = 0.14 for Fisher-layer
> selection — linear separability varies with model scale. The smoke run on
> 1B will be diagnostic: if gate AUC < 0.70 at 1B, we move to 2B before
> reporting failure.

**Q: Does the refusal-subspace projection lock (Guard A, Rogue Scalpel) interact
with the CAST condition vector?**

> Guard A projects behavior vectors out of the refusal-formation subspace S.
> The condition vector is never written into the residual stream — it is read-only —
> so Guard A does not apply to it. Guard E (conditional gating) IS this experiment.

---

## 10. Verification Checklist

- [ ] Smoke run on Gemma-3-1B-it passes (gate AUC > 0.60, harmful refusal > 30%)
- [ ] Condition vector extracted at L_c; behavior vector at L_b; L_c < L_b confirmed
- [ ] DiffMean extraction: 50 harmful + 50 harmless contrast pairs (fixed seed)
- [ ] Theta_c grid search recorded in experiment_log.jsonl (7 alpha × 4 coeff combos)
- [ ] Harmless refusal < 3% at >= 1 operating point confirmed at 3-seed median
- [ ] Harmful refusal delta >= 50 pp confirmed at same operating point
- [ ] MMLU delta on benign set recorded (five-axis composite logged)
- [ ] PPL on harmless set recorded and below cliff
- [ ] Gate PR-AUC logged; comparison to unconditional PR-AUC recorded
- [ ] Norm-budget offshell displacement Delta||h|| logged (N5 compliance)
- [ ] JailbreakBench Compliance Rate reported (Rogue Scalpel mandate)
- [ ] Row updated in IDEA_TABLE.md; result in EXPERIMENT_LEDGER.md

---

## 11. Status Journal

(Append-only)

- 2026-05-31 — Design doc created. Status: UNTESTED / PENDING. Code needed:
  (a) CAST gate module (`src/steering/cast.py`: condition extraction + threshold
  gate + behavior write dispatch); (b) dual-forward probe infrastructure
  (`src/steering/guard.py`: Guard D verdict check); (c) JailbreakBench-100
  harmful prompt subset and AlpacaEval-100 benign subset, fixed seed, wired
  into eval bundle. E1, E4, E7 results (SUPPORTED) provide the extraction
  parameters; E3 cliff result constrains the alpha grid. Blocked on CAST code.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-B (steering research critic). Critiques the idea, not
the implementation.*

### Prior plausibility (independent of the CAST framing)

MEDIUM-HIGH. The core claim — that gating on input context reduces false
positives of safety steering — is well-motivated mechanistically. The
condition-vector mechanism is the same linear operation as behavior extraction;
the only question is whether the separability of harmful vs harmless inputs at
early layers of Gemma-2-2B is sufficient to support a 3% harmless-refusal
target. That is an empirical question the protocol answers.

### Mechanism scrutiny

The mechanism is clear: read-only dot product at L_c gates write at L_b.
The "because" clause is: harmful inputs place h_early closer to the harmful
mean (by construction of DiffMean), so cos(h, v_condition) is systematically
higher for harmful inputs. This is the Fisher-discriminant argument, and E2's
failure at Gemma-270m is a real concern. The gap between 270m and 2B scale
needs explicit justification in the smoke run.

### Confounds

1. **Threshold instability:** theta_c that achieves 3% harmless refusal on the
   evaluation set may not generalize to OOD inputs — E15 directly tests this.
2. **Prompt-set contamination:** if harmful and harmless evaluation prompts share
   surface tokens, the condition vector may be tracking vocabulary rather than
   semantics, conflating with a token-level classifier.
3. **Alpha dependence:** the claimed 50 pp uplift is at a specific alpha; if
   alpha must be high to reach 50 pp, the harmless-refusal constraint may be
   impossible simultaneously.

### Numerology check — does the 3% / 50 pp threshold specifically matter?

The 3% harmless-refusal threshold is a practical deployment target (XSTest
convention), not a fundamental number. The 50 pp uplift is generous — even
30 pp would be useful. The pre-registered thresholds are tight enough to be
falsifiable but should be interpreted as a success criterion, not a
theoretical prediction.

### Literature: precedent or rediscovery?

CAST (arXiv:2409.05907) is the direct precedent. This experiment is a
reproduction + extension to Gemma-2-2B with an explicit harmless-refusal
target. It is not a novel method but a falsifiable reproduction run that either
validates or challenges CAST's reported results on a different architecture.

### Expected effect size — skeptical a-priori re-prediction

My prior: the harmless-refusal target of < 3% is achievable at the cost of
some harmful-refusal degradation (the gate may miss borderline harmful inputs).
The joint target (< 3% harmless AND >= 50 pp harmful uplift) is harder. 90%
CI for harmless-refusal at optimal theta_c: [1%, 8%]; for harmful-refusal
uplift: [35 pp, 65 pp]. The joint success probability under my prior is ~40%.

### Minimum-distinguishing experiment

Unconditional steering at alpha producing 50% harmful refusal, record harmless
refusal. Then add CAST gate. Two conditions, 3 seeds each, one model.
Wall-clock: ~45 min on a 4090. This is already the primary protocol (Section 7.1).

### Verdict

**NOVEL+TESTABLE** in the Gemma-2-2B context (genuine reproduction gap).
The mechanism is principled, the falsifier is specific, and the protocol is
minimal. Main risk: linear separability at early Gemma-2-2B layers for the
harmful/harmless contrast; the smoke run on 1B is the diagnostic gate.

---

## Pseudocode & Methodology

This is the **CAST baseline** for Block B: a read-only **condition vector** at an early
layer `L_c` gates a refusal **behavior write** at a later layer `L_b`. The knob varied is
the **condition threshold theta_c** (with a secondary alpha grid). Two vectors are built;
the gate adds zero norm budget (METHODOLOGY §2 — it is a read, not a write).

### 1. Steering-vector recipe

```python
# CONDITION vector: DiffMean of EARLY-layer activations on harmful vs harmless prompts.
# DiffMean (not PCA) is justified by E4 (cos 0.994-0.996). METHODOLOGY §1.3.
Hc = collect_activations(model, tok, harmful_vs_harmless_pairs, layer=L_c)   # L_c in [6,12]
v_condition = diffmean_vector(Hc.pos, Hc.neg)        # mean(harmful)-mean(harmless) @ L_c

# BEHAVIOR vector: the refusal direction, DiffMean at a LATER write layer.
Hb = collect_activations(model, tok, refuse_vs_comply_pairs, layer=L_b)      # L_b in [14,20]
v_behavior  = diffmean_vector(Hb.pos, Hb.neg)        # refusal direction @ L_b
assert L_c < L_b                                      # required ordering (E26 tests reverse)
```

### 2. Experiment procedure

```text
1. Extract v_condition @ L_c and v_behavior @ L_b (50 harmful / 50 harmless pairs; E1).
2. For each eval prompt, read h_early @ L_c and compute the gate (CosineGate):
3.     gate = ( cos(h_early, v_condition) > theta_c )           # read-only, zero budget
4. Behavior write fires ONLY where the gate fired:
5.     h_late' = h_late + alpha * gate * v_behavior             # apply_operation 'add', masked
6. for theta_c in {0.1,0.2,0.3,0.4,0.5,0.6,0.7}:                # the ONE knob
7.     for alpha in {5,10,15,20}:                               # relative alpha (E7)
8.         measure harmful-refusal rate, harmless-refusal rate,
                  MMLU (benign), PPL (harmless), gate PR-AUC.
9. COMPARE against UNCONDITIONAL steering at the same alpha (gate always on).
10. Eval set: 100 harmful (JailbreakBench) + 100 benign (AlpacaEval), fixed seed.
```

### 3. Measurement & decision rule

PRIMARY metrics: harmless-input refusal rate and harmful-input refusal uplift at the
best theta_c. FALSIFIER (§3): DISCARDED if, after the theta_c grid search, NO threshold
simultaneously achieves harmless-refusal `< 3%` AND harmful-refusal delta `>= 50 pp`
above the no-steer baseline (3-seed median); ALSO DISCARDED if the optimal CAST point
does not Pareto-dominate unconditional steering on the (harmful-refusal, harmless-refusal)
plane. Pre-registered (§6): harmless-refusal `< 3%`; harmful-refusal `>= 50 pp` uplift;
gate AUC `[0.80, 0.92]`; composite improvement `[+0.05, +0.15]` vs unconditional.
Verdict: SUPPORTED iff both core thresholds met at one operating point AND CAST
Pareto-dominates unconditional. The smoke run on Gemma-3-1B is diagnostic: if gate
AUC `< 0.70` there, move to 2B before reporting failure (E2's rho=0.14 separability
caveat).

### 4. Where the code is / status

The gate primitive **exists**: `src/steering/gate.py` provides `CosineGate` (and
`evaluate_gates` / `pr_auc` / `roc_auc`) plus `condition_features`. What is **missing**
(why E9 is UNTESTED) is the full CAST *pipeline* that wires a read-only `L_c` condition
into a masked `L_b` write inside one forward pass — i.e. a `cast_steer` dispatch around
`hooks.SteeringContext`, the JailbreakBench-100 / AlpacaEval-100 eval subsets, and the
Guard-D dual-forward verdict check. The condition-vector math reuses `extract.diffmean_vector`
and the gate-quality math reuses `gate.pr_auc`; the dispatch glue and the safety eval
bundle are the new machinery. UNTESTED, blocked on this CAST gating pipeline.

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/E9.md`.
