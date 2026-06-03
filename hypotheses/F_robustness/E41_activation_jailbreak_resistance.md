# E41 — Activation-Based Conditional Refusal Resists Jailbreak Suffixes

> **One-line claim:** Conditional refusal steering (CAST-style, reading
> activations not tokens) resists adversarial jailbreak suffixes better
> than system-prompt refusal, because the condition reads the semantic
> activation — which suffix obfuscation cannot fully corrupt — not the
> surface token sequence that suffixes target.
>
> **Block:** F — Robustness, safety, and evaluation (E41-E50).
> **Primary axis:** A5 (WHEN — condition/gate).
> **Implementation status:** `o planned / UNTESTED`.

---

## 1. Motivation (>= 100 words)

Adversarial jailbreak suffixes (Zou et al. 2023 GCG; arXiv:2307.15043)
are token sequences appended to harmful prompts that fool a language model
into complying by disrupting the refusal circuit at the token-processing
level. These suffixes work by finding token sequences whose embeddings,
when accumulated in the residual stream, push the model off the refusal
ridge before the refusal direction can assemble. System-prompt refusal
("You must not help with...") is particularly vulnerable because the system
prompt is itself part of the token sequence that the suffix can be designed
to counteract. Conditional refusal steering (CAST-style, Block B of our
program) operates differently: the condition is a read on the hidden-state
projection <h, v_condition> at a mid-layer, where v_condition is the
"this is a harmful request" direction extracted from contrast pairs. The
condition check occurs AFTER the early-layer token processing, at a point
in the residual stream where the semantic content of the request has been
assembled. The key insight is that adversarial suffixes primarily corrupt
the token-level representation (they are designed to fool the attention-
based token routing), but the mid-layer semantic direction — especially
the "harmfulness assessment" direction — may be harder to corrupt because
it is an aggregate of many attention heads' contributions. The Rogue
Scalpel paper (arXiv:2509.22067) confirmed that the refusal mechanism
sits on a fragile mid-layer ridge; the question is whether a sufficiently
adversarial suffix can corrupt the CONDITION projection as well as the
refusal ridge itself. If the condition projection is more robust than the
refusal ridge, CAST-style gating provides a second layer of defense:
even if the suffix disrupts the native refusal, the CAST gate can detect
that the request is harmful (via the activation condition) and apply a
refusal steer. Our Rogue-Scalpel-direction screening (Compliance Rate
0.80 -> 1.00 under steering on Gemma-3-270m, FINDINGS.md S-4, S-8)
establishes that the baseline steering setup does interact with safety;
E41 specifically tests whether the CAST gate is more robust than the
baseline.

---

## 2. Formal Hypothesis (>= 50 words)

**H:** On Gemma-2-2B-it, CAST-conditioned refusal steering will maintain
a harmful-request refusal rate >= 80% under GCG-style adversarial suffix
attacks (from JailbreakBench), while a system-prompt-only refusal baseline
(no activation-based condition) will show refusal rate degradation to <
50% under the same attacks. The activation-based condition probe will
retain a precision >= 0.70 for detecting harmful requests even in the
presence of adversarial suffixes, because the condition reads the semantic
representation not the surface tokens.

---

## 3. Falsifier (>= 30 words)

If CAST-conditioned refusal achieves < 70% refusal rate under adversarial
suffixes (i.e., the adversarial suffixes successfully corrupt the activation-
based condition as well as the refusal ridge), the hypothesis is DISCARDED
(Status `x disproved`). The key falsifier is that the activation condition
is not meaningfully more robust than the token-based system prompt.

---

## 4. Citations (Citation-Rigor format, >= 80 words)

```
Zou, Andy, et al. 2023 'Universal and Transferable Adversarial Attacks on
Aligned Language Models' arXiv:2307.15043 — GCG (Greedy Coordinate Gradient)
jailbreak attack; generates adversarial suffixes that cause refusal bypass;
the attack methodology used to construct E41's test suffixes from the
JailbreakBench suite.

Chao, Patrick, et al. 2024 'JailbreakBench: An Open Robustness Benchmark
for Jailbreaking Large Language Models' arXiv:2404.01318 — JailbreakBench;
the 100-prompt, 10-category safety evaluation used by Rogue Scalpel as
the primary safety benchmark; provides the adversarial prompts for E41.

Korznikov, et al. 2025 'The Rogue Scalpel: Activation Steering Compromises
LLM Safety' arXiv:2509.22067 — Rogue Scalpel; the universal-attack (F5)
construction (average 20 jailbreaking vectors) is the most relevant attack
for E41; its Guard Layer E (conditional gating) is the defense tested here.

Turner, et al. 2023 'Activation Addition: Steering Language Models Without
Optimization' / Rimsky, Nina, et al. 2023 'Steering Llama 2 via Contrastive
Activation Addition' arXiv:2312.06681 — CAA; the DiffMean-based condition
and behavior vector extraction methodology.
```

---

## 5. Mechanism

The CAST-style gate works as follows (from the first-principles corpus,
Part 1 Step 4):

    layer k:  read s = <h_k, v_condition> / ||h_k||
              if s > theta: apply alpha * v_refusal at layer m > k
              else: pass through unchanged

A GCG suffix appends tokens that, when processed by the model, modify
the residual stream in a direction that suppresses the native refusal
assembly. The suffix is optimized against the TOKEN-SPACE gradient of
the harmful-compliance probability, which primarily targets the attention-
score routing and embedding-layer token processing.

The activation-based condition probe reads the RESIDUAL STREAM at a mid-
layer, after the early-layer token processing has integrated the suffix.
Two questions: (1) does the GCG suffix also corrupt the v_condition
projection at the mid-layer? (2) Even if partially corrupted, does the
condition projection remain above theta more reliably than the native
refusal mechanism does above its own threshold?

If the v_condition direction is a robust aggregate of semantic "harmfulness"
signal assembled from many attention heads, it may be harder to suppress
with a token-level suffix than the single-direction refusal ridge (which
the Rogue Scalpel paper showed is fragile). The experiment measures both
the condition-probe precision under suffix attack and the end-to-end
refusal rate of the gated system.

The Rogue Scalpel Guard Layer D (dual-forward verdict check) is also
relevant: if the condition probe is defeated but the CAST gate is combined
with the verdict check, the combined defense may still hold.

---

## 6. Predicted Delta

| Condition | Refusal rate under GCG suffixes |
|---|---|
| No defense (base model) | 0-20% (model is jailbroken) |
| System prompt only | 20-50% (partially resistant) |
| CAST condition gate | >= 70% (hypothesis prediction) |
| CAST + Rogue Scalpel Guard D | >= 80% |

Condition probe precision (detecting harmful requests under suffix attack):
>= 0.70 at the primary CAST threshold.

---

## 7. Protocol

### 7.1 Primary experiment

- Model: Gemma-2-2B-it (4-bit), RTX 4090.
- Attack: JailbreakBench 100-prompt suite (10 categories); GCG-style
  adversarial suffixes (from the JailbreakBench repository or generated
  with a suffix-optimization tool on a held-out set of prompts, not the
  eval set). Target: 20 prompts for suffix generation, 80 for evaluation.
- Conditions: (A) no defense; (B) system prompt refusal only; (C) CAST
  condition gate (condition layer k, behavior injection layer m); (D) CAST
  + Guard D (dual-forward verdict check).
- Condition vector: DiffMean "harmfulness" direction from Sorry-Bench
  contrast pairs (harmful vs harmless). Condition layer: identified from
  prior E2/layer-sweep or set at the refusal-circuit-active layer per
  arXiv:2604.08524.
- Eval: refusal rate (LLM-as-judge, Qwen3-8B or similar at 94% precision
  calibration per Rogue Scalpel); JailbreakBench CR; MMLU-500 (confirm
  gate does not cause over-refusal on benign inputs — compare with E42).
- Seeds: 3 (screening), 7 for rung-3.
- Baseline CR: confirm 0% on the baseline unsteered model before attack.

### 7.2 Where it shines

This experiment directly tests the safety value proposition of CAST-style
activation-based gating under realistic adversarial conditions. It is the
robustness validation of Block B's conditional steering family.

---

## 8. Cross-references

- IDEA_TABLE.md Block F row E41.
- E9 (CAST harmless refusal gate): E41 is the adversarial robustness test
  of E9's gating approach; E9 tests in-distribution, E41 tests under attack.
- E47 (gate + ortho-stack + norm-cap combination): E47 assembles the full
  guard stack; E41 tests the gate component specifically.
- Rogue Scalpel Guards A-E (arXiv:2509.22067): E41 tests Guard Layer E
  (CAST conditional gate) as the primary robustness mechanism.
- FINDINGS.md S-4, S-8: Rogue-Scalpel-direction screening (CR 0.80->1.00
  under steering on Gemma) — establishes baseline safety interaction;
  E41 specifically tests the guard as a defense.
- arXiv:2307.15043 (GCG): the attack methodology.
- arXiv:2404.01318 (JailbreakBench): the evaluation suite.

---

## 9. Committee Q&A

**Q: GCG suffixes are computationally expensive to generate. Can the
experiment use pre-generated suffixes from the JailbreakBench repository?**

> Yes; JailbreakBench provides pre-computed GCG suffixes for standard
> models. However, Gemma-2-2B is not the primary target of those suffixes,
> so their transfer effectiveness must be verified first. If transfer is
> < 20% CR increase on the base model (unguarded), generate Gemma-specific
> suffixes on the held-out prompt set.

**Q: Could the CAST gate cause over-refusal on benign suffixed prompts?**

> This is the E42 concern (over-refusal on benign look-alikes). E41 reports
> MMLU and XSTest refusal rates as secondary metrics; if the gate's
> precision drops under suffix attack (many benign suffixed prompts trigger
> the gate), the over-refusal cost must be reported alongside the
> harmful-refusal gain.

**Q: Isn't this just showing that activation-based detection is better than
prompt-based detection? That seems obvious.**

> The claim is non-obvious because adversarial suffixes are specifically
> designed to corrupt activation-space representations, not just token-space
> representations. GCG targets the activation gradient; whether the condition
> direction is harder to corrupt than the refusal direction is an empirical
> question. The Rogue Scalpel's F4 finding (no stable direction screen) makes
> this non-trivial.

---

## 10. Verification checklist

- [ ] JailbreakBench prompts loaded; baseline CR on unguarded model confirmed
      0% before any suffix or steering.
- [ ] GCG suffix transfer effectiveness verified on Gemma-2-2B; if < 20%
      CR increase, generate Gemma-specific suffixes.
- [ ] CAST condition vector extracted from Sorry-Bench pairs, >= 50 contrast
      pairs; condition layer pre-specified.
- [ ] LLM-as-judge calibrated on a 50-example human-annotated slice
      (harmful/benign; >= 90% agreement with human labels required).
- [ ] MMLU-500 and XSTest benign-refusal rates reported for conditions A-D.
- [ ] Rung-3 gate: n >= 7, Wilcoxon + bootstrap CI (Holm corrected).
- [ ] IDEA_TABLE.md row updated; FINDINGS.md S-4/S-8 screening observations
      cross-referenced in status journal.

---

## 11. Status Journal

- 2026-05-31 — Created from corpus block F, hypothesis E41.
  Status: `o UNTESTED`.

  Cross-ref to screening: FINDINGS.md S-4 (Gemma-3-270m CR 0.80->1.00
  under steering) and S-8 (Gemma-3-1b same pattern) document the Rogue-
  Scalpel-direction finding — ANY steering of the Gemma models in our
  harness with sufficient alpha raises CR. E41 specifically tests whether
  the CAST GATE prevents this CR rise when the input is genuinely harmful.
  These are distinct questions: the screening result shows the RISK; E41
  tests the MITIGATION.

  Dependency: JailbreakBench infrastructure, LLM-as-judge calibration,
  CAST gate implementation (Block B tooling).

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-F (adversarial robustness + safety specialist).*

### Prior plausibility
**MEDIUM-HIGH.** The conceptual argument (activation-based condition reads
semantic content, suffix corrupts token routing) is plausible. But GCG is
specifically designed to optimise LOSS GRADIENTS which flow through the
activation space, so the suffix does corrupt activations — the question
is whether it corrupts the specific direction v_condition or only the
refusal ridge direction.

### Mechanism scrutiny
The mechanism correctly identifies that GCG targets the token-gradient, but
GCG's loss is computed on the final output probability, which propagates
gradients through ALL activation layers including the condition-probe layer.
So GCG could in principle also corrupt the condition projection. The key
empirical question is whether v_condition is a more robust direction (higher
margin) than the refusal ridge, not whether GCG doesn't touch activations.

### Confounds
1. Adaptive attack: if the adversary knows that a CAST gate is deployed,
   they could generate suffixes that optimise against the condition projection
   specifically. E41 should include an adaptive-attack condition (optimize
   suffix loss against condition probe + refusal together) as a worst-case.
2. The 80-20 split (20 for suffix generation, 80 for eval) may lead to
   over-fitting if the suffix generation model uses the same condition
   direction that the defense uses.

### Expected effect size
My prior: CAST gate achieves 60-75% refusal rate under standard GCG
(not >= 80%), because GCG suffixes do partially corrupt mid-layer activations.
The Guard D (verdict check) addition may push this to 75-85%.

### Verdict
**TESTABLE + SAFETY-CRITICAL** — This is one of the highest-priority
experiments in the program given the Rogue Scalpel threat model. Recommend
including the adaptive-attack condition and reporting all five measurement
axes (behavior, capability, coherence, safety, selectivity).

---

## Pseudocode & Methodology

This section specializes [`../METHODOLOGY.md`](../METHODOLOGY.md) to E41. E41 is
a **safety/robustness** test: a CAST-style activation condition gate vs a
system-prompt baseline under GCG adversarial suffixes, measured by JailbreakBench
compliance.

### 1. Steering-vector recipe (two DiffMean directions: condition + refusal)

```python
# §1.3 METHODOLOGY: both directions are closed-form DiffMean (NO training).
# (a) The CONDITION direction (the gate's "this is a harmful request" reader):
bank_c = extract.build_vector_bank(model, tok,
            load_concept("harmfulness"), layer_k)        # Sorry-Bench harmful vs harmless
v_condition = bank_c[layer_k]["diffmean"]                # mean(h_harmful) - mean(h_harmless)
v_condition = v_condition / norm(v_condition)            # unit (gate reads a normalised projection)

# (b) The REFUSAL behavior direction (what the gate applies when it fires):
bank_r = extract.build_vector_bank(model, tok, load_concept("refusal"), layer_m)
v_refusal = bank_r[layer_m]["diffmean"]                  # layer_m > layer_k (read early, write later)

# extract.best_layer (fisher_ratio) chooses layer_k where harmful/harmless separate cleanly.
```

### 2. Experiment procedure (4 conditions x clean/suffixed prompts)

```text
1. CONFIRM baseline: JailbreakBench CR = 0% on the unsteered model (§10).
2. Build attacks: GCG suffixes (Zou 2023) for 20 held-out prompts; eval on 80.
   Verify Gemma-2-2B suffix transfer; if < 20% CR rise, regenerate Gemma-specific.
3. Conditions: (A) no defense; (B) system-prompt refusal only;
   (C) CAST gate; (D) CAST gate + Guard D (dual-forward verdict check).
4. CAST gate logic (the §5 mechanism) applied per forward pass:
       s = <h_k, v_condition> / ||h_k||                  # read at layer_k
       if s > theta:  hooks.apply_operation(h_m, v_refusal,
                          operation="add", alpha)         # h' = h + alpha*v_refusal
       else:          pass through unchanged              # gate does NOT fire
   (theta calibrated on a held-out 25-harmful/25-benign set, not the eval set.)
5. MEASURE (§3 METHODOLOGY): refusal rate under suffix attack (off-family judge);
   JailbreakBench CR; condition-probe precision under attack; MMLU-500 + XSTest
   (over-refusal dual, vs E42); PPL; geometry probes.
```

### 3. Measurement & decision rule

- **PRIMARY metrics:** harmful-request refusal rate under GCG suffixes for the
  CAST gate (condition C) and the condition-probe precision.
- **Hypothesis (§2):** CAST refusal rate >= 80% under attack (vs system-prompt
  baseline degrading to < 50%); condition-probe precision >= 0.70 under suffix.
- **Pre-registered FALSIFIER (§3):** if CAST-conditioned refusal achieves < 70%
  refusal rate under adversarial suffixes (the suffix corrupts the activation
  condition as well as the refusal ridge), DISCARD (`x disproved`).
- **Verdict logic:** a safety leak (CR rise on harmful prompts) under steering is
  an automatic DISCARD regardless of behavior score (§10 mandate); report all five
  axes including XSTest over-refusal so a refusal gain is never bought with
  over-refusal.

### 4. Where the code is / status — UNTESTED

- **No driver yet.** A campaign would log rows; then `scripts/build_provenance.py`
  generates `hypotheses/PROVENANCE/E41.md`.
- **Missing machinery (why UNTESTED):** (a) a **CAST gate** that reads
  `<h_k, v_condition>` and conditionally applies `v_refusal` at `layer_m` (Block B
  tooling — the per-position conditional hook does not yet exist in `hooks.py`);
  (b) a **GCG suffix generator / loader** plus Gemma-2-2B transfer verification;
  (c) a **calibrated safety judge** at >= 90% human agreement for refusal
  detection; (d) JailbreakBench + XSTest dataset wiring; (e) Guard D (dual-forward
  verdict check) for condition D. Screening context: FINDINGS.md S-4/S-8 show ANY
  Gemma steering raises CR 0.80->1.00 (the RISK); E41 tests the gate as the
  MITIGATION — a distinct question.

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/E41.md`.
