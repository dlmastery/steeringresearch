# E8 — Instruction-Tuned to Base Model Transfer

> **One-line claim:** Steering vectors extracted from an instruction-tuned Gemma model
> transfer to the corresponding base model with <20% efficacy loss, indicating that
> behavioral directions are established during pretraining and merely amplified by
> instruction tuning.
>
> **Source design space:** Block A — Foundations and measurement tooling (E8).
> **Primary axis:** A7 (HOW DERIVED — extraction source model / training regime).
> **Implementation status:** `o UNTESTED`.

---

## In Plain English

**What we're testing, simply:** Models come in two versions: a raw "base" version,
and an "instruction-tuned" version that's been polished to follow instructions and
refuse harmful requests. This asks whether a steering nudge built from the polished
version still works on the raw version — which would tell us where these behaviors
actually live inside the model.

**Key terms (defined here so you don't have to look anything up):**
- **Language model (LLM):** an AI that predicts the next word; here, small Gemma
  models.
- **Steering:** nudging the model's behavior by adding a direction to its internal
  state while it writes.
- **Steering vector:** the specific direction of the nudge.
- **Residual stream:** the model's running internal state, where the nudge is added.
- **Layer:** one of the model's stacked processing steps.
- **DiffMean:** the simple recipe for building the nudge.
- **alpha (strength):** how hard we push the nudge.
- **Pretraining:** the first, huge training phase where the model learns language
  from raw text.
- **Base model:** the model straight out of pretraining — not yet polished.
- **Instruction-tuned (IT) model:** the base model after extra polishing that
  teaches it to follow instructions and refuse harmful requests.
- **Transfer:** taking a nudge built on one version and applying it to the other.
- **Efficacy loss:** how much weaker the nudge gets when transferred. Under 20% loss
  counts as "transfers well."

**Why we're doing this (the point):** If a nudge built on the polished model also
works on the raw model, it means these behaviors were already learned during the
big pretraining phase, not just bolted on by the polishing. Practically, the
cheaper-to-run base model could reuse the polished model's safety nudges.

**What the result would mean:** A positive result means behavioral directions are
baked in during pretraining and are portable across versions. A negative result —
especially for a safety/refusal nudge — would mean those behaviors come from the
polishing step and don't exist in the raw model.

See [`../GLOSSARY.md`](../GLOSSARY.md) for any other term.

---

## 1. Motivation (>=100 words)

Instruction-tuned (IT) models are aligned versions of base pretrained models: they
receive RLHF, DPO, or supervised fine-tuning (SFT) on demonstration data that teaches
the model to follow instructions, refuse harmful requests, and maintain specific personas.
The question E8 addresses is: are the behavioral concept directions that steering exploits
a product of pretraining, or of instruction tuning? If they are pretraining-induced, then
a steering vector extracted from an IT model should transfer to the base model with minimal
efficacy loss — because the direction exists in both models' activation spaces, differing
only in the strength of the behavioral signal (the dot product h · v_behavior may be
smaller in the base model, but the direction v should still exist). If behavioral directions
are primarily instruction-tuning-induced, the base model's activation space would lack a
corresponding direction, and the IT vector would either have no effect or cause incoherence
when injected. This distinction matters practically: base models are cheaper to run
(no alignment tax) and have more capacity for novel behaviors; if they can be steered
by IT-derived vectors, the IT model's alignment work can be "donated" to base deployment.
It also matters theoretically: it bears on the Linear Representation Hypothesis's scope —
does pretraining alone give rise to behaviorally interpretable linear directions in
activation space? The Rogue Scalpel paper (arXiv:2509.22067) found that its steering
attacks worked on both base and IT models, suggesting the refusal direction is at least
partially pretraining-induced. E8 quantifies this systematically.

---

## 2. Formal hypothesis (>=50 words, falsifiable)

For Gemma-2-2B-it (instruction-tuned) and Gemma-2-2B (base), both evaluated at the
same layer L16: a DiffMean steering vector extracted from the IT model, when applied
to the base model at the same relative alpha (E7 parameterization), produces behavior
success >= 80% of the behavior success achieved by the same vector in the IT model
(i.e., < 20% efficacy loss). The reverse direction (base vector in IT model) is a
secondary test with no pre-registered threshold. Both directions use a generation-based
judge on the same prompts.

---

## 3. Falsifier (>=30 words)

**Fired if:** the IT-to-base efficacy loss exceeds 20% on more than one tested behavioral
concept (at matched relative alpha), when both vectors are unit-normalized and the
generation judge is calibrated identically. A loss exceeding 20% for a SAFETY concept
specifically (refusal steering) would be especially significant, as it would imply the
refusal direction is instruction-tuning-induced, not pretraining-induced.

---

## 4. Citations (>=80 words, Citation Rigor format)

```
Panickssery, Aryan, et al. 2023 arXiv 'Steering Llama 2 via Contrastive
Activation Addition' (arXiv:2312.06681) — CAA extracts vectors from IT
models and does not test base-model transfer; E8 fills this gap using
their extraction methodology.

Korznikov, Mikhail, et al. 2025 ICML 'The Rogue Scalpel: Activation
Steering Compromises LLM Safety' (arXiv:2509.22067) — Rogue Scalpel
showed that steering-based jailbreaks work on IT models because the
refusal direction can be steered away; E8 asks whether this direction
is pretraining-induced by testing IT→base transfer.

Li, Kenneth, et al. 2023 NeurIPS 'Inference-Time Intervention: Eliciting
Truthful Answers from a Language Model' (arXiv:2306.03341) — ITI's
"truthfulness" directions were extracted from InstructGPT; E8 asks whether
the same directions would work in a pretrained-only GPT; the architecture
difference prevents exact replication but motivates the hypothesis.

Zou, Andy, et al. 2023 NeurIPS 'Representation Engineering: A Top-Down
Approach to AI Transparency' (arXiv:2310.01405) — RepE uses principled
activations (instruction pairs) to probe representational geometry; their
claim that representations reflect model "knowledge" predicts that
pretraining establishes the directions E8 tests.

Chalnev, Aleksandr, et al. 2024 arXiv 'Refusal in Language Models Is
Mediated by a Single Direction' (arXiv:2406.11717) — finds a single
refusal direction in IT models; does not test base-model transfer; E8
tests whether this direction exists in the base model.
```

---

## 5. Mechanism (deep technical)

Pretraining on large text corpora instills statistical regularities about the world,
including co-occurrences of behaviors with contexts. For example, "I cannot help with
that" appears near harmful requests in pretraining data; "Voilà" appears near French
text. The Linear Representation Hypothesis (LRH) predicts these regularities are encoded
as linear directions in h before instruction tuning. Instruction tuning amplifies these
directions by reinforcing them via RLHF/DPO, but does not create them from scratch —
it steers the model to USE the pretraining-induced directions more reliably.

Mechanism predictions by concept type:

1. **Safety/refusal concepts**: the refusal direction (arXiv:2406.11717) is likely
   pretraining-induced because harmful/safe contexts are well-represented in pretraining
   data (news, fiction, internet text all contain this contrast). Transfer to base model
   expected to work (efficacy loss < 20%).

2. **Stylistic concepts (language, formality)**: the French/formal/informal direction is
   almost certainly pretraining-induced (multilingual text in pretraining). Transfer to
   base expected to work even better than safety.

3. **Behavioral concepts (sycophancy avoidance, honesty)**: these may be more instruction-
   tuning-specific if the behavior was primarily shaped by RLHF rather than pretraining
   data statistics. Transfer to base model may fail here (efficacy loss > 20%).

The IT-to-base geometry: both models share the same tokenizer and the first layer's
embedding matrix (which is frozen during IT in many setups). The hidden state h at
layer L in the IT model is a function of the same weights up to the fine-tuning deltas
(typically LoRA or small SFT updates applied to attention/MLP). If the LoRA rank is
small (r=8–32 in typical Gemma instruction tuning), the weight changes are low-rank
and the base model's residual stream geometry is approximately preserved in most
subspaces. The steering direction (in the high-dimensional space orthogonal to the
LoRA update subspace) should therefore transfer.

Transfer failure mechanism: if the IT model's LoRA updates rotate the activation space
near the behavioral concept direction (i.e., the LoRA is aligned with v_behavior), then
the IT-extracted vector v lives in a subspace that is rotated in the base model, and
the cosine alignment fails. This is testable: measure cos(v_IT, v_base) where v_base
is the DiffMean vector extracted from the base model on the same contrast set; if
cos > 0.90, the directions align and transfer should work.

The base-to-IT direction (secondary test): steering the IT model with a base-extracted
vector is expected to also work (and possibly work better on capability-aligned behaviors
since the base model's directions are not filtered by alignment training).

---

## 6. Predicted Delta (pre-registered numbers)

| Metric | Predicted value | Observed |
|--------|----------------|---------|
| IT→base efficacy loss (safety concept) | < 20% | **UNTESTED** |
| IT→base efficacy loss (stylistic concept) | < 10% | **UNTESTED** |
| IT→base efficacy loss (behavioral/honesty) | 10–40% (uncertain) | **UNTESTED** |
| cos(v_IT, v_base) at same layer | >= 0.90 | **UNTESTED** |
| base→IT efficacy (secondary) | >= 70% of IT→IT | **UNTESTED** |

---

## 7. Experimental protocol

### 7.1 Primary experiment

- **Model pair:** Gemma-2-2B-it (source: vector extraction) and Gemma-2-2B (base:
  target model for transfer). Both loaded at the same layer (L16).
- **Concepts:** 3 behaviors — (i) refusal/safety (harmful vs harmless contrast),
  (ii) French vs English (stylistic), (iii) sycophancy avoidance (behavioral).
- **Extraction:** DiffMean on 50+ contrast pairs from the IT model on each concept.
- **Injection:** same relative alpha (E7 parameterization: alpha_rel = 0.10) in both
  IT and base models, using the IT-extracted vector.
- **Metrics:** behavior success via LLM-judge (same judge prompt for both models to
  ensure comparability); PPL on WikiText; JailbreakBench CR.
- **Secondary:** extract v_base from the base model on the same concept; compare
  cos(v_IT, v_base); inject v_base into IT model; record efficacy.
- **Seeds:** n=7 (rung-3 gate).

### 7.2 Where this should SHINE

Concepts with a pretraining-strong prior: multilingual behavior (language direction is
pretraining-universal), factuality/truthfulness (well-represented in pretraining data),
and safety concepts (harmful/safe text is ubiquitous in pretraining).

### 7.3 Where it may FAIL

Concepts that require instruction-tuning-specific behaviors: task-following conventions
(e.g., "respond with numbered lists"), RLHF-specific verbal patterns, or any concept
where the positive examples are predominantly from instruction-tuning data. These are
expected to have lower transfer (> 20% efficacy loss) and should be noted as an
IT-specific behavior.

### 7.4 Practical significance

If IT→base transfer holds for safety vectors, a safety-critical use case emerges:
extract a "refusal boost" vector from the IT model and inject it at inference time
into a base model that has been fine-tuned for a specialized task (where the full IT
alignment would reduce task performance). This is the "alignment donation" pattern.

---

## 8. Cross-references

- **E4** (DiffMean ~= PCA): the high cos is expected between IT-extracted DiffMean
  and IT-extracted PCA; E8 adds a third model (base) to the alignment triangle.
- **E7** (relative alpha): essential for the IT→base comparison (‖h‖ may differ
  between base and IT models for the same input).
- **Refusal direction** (arXiv:2406.11717): the single refusal direction found in IT
  Gemma is the primary concept for E8's safety test.
- **Rogue Scalpel** (arXiv:2509.22067): the attack worked on IT models; E8 tests
  whether it would also work on base models via IT-extracted vectors.
- **Non-Identifiability** (arXiv:2602.06801): the large equivalence class argument
  suggests IT-derived directions are likely to be in the same class as base directions.
- **N4** (steering as inverse ICL): ICL demonstrations in the IT model produce
  activation deltas aligned with v_IT; in the base model, the same ICL demonstrations
  should produce v_base; if v_IT ~ v_base, N4 holds in both.
- **IDEA_TABLE.md** Block A row E8.

---

## 9. Committee Q&A

**Q: How do you handle the fact that base Gemma and IT Gemma may produce different
outputs for the same prompts (the alignment changes the distribution)?**

> The generation judge must be calibrated for each model separately (baseline behavior
> without steering). The efficacy metric is the CHANGE from the model's own baseline:
> (steered_success - baseline_success) / baseline_success for each model. This makes
> the comparison fair even if the IT model has a higher baseline refusal rate than the
> base model.

**Q: Gemma-2-2B base is not gated in the same way as Gemma-2-2B-it — can you load them
from the same repo?**

> Both are gated on HuggingFace behind the Gemma license. The base and IT variants
> are separate model checkpoints (google/gemma-2-2b and google/gemma-2-2b-it) but share
> the same license acceptance. They can be loaded side-by-side in fp16 mode (~9 GB total)
> on the 16 GB GPU.

**Q: If the IT model has been fine-tuned with LoRA, doesn't the LoRA specifically
modify the layers where the behavioral direction lives, making transfer unlikely?**

> Gemma-2-2B-it is fine-tuned with full-weight SFT + DPO (not LoRA) per the Gemma
> 2 technical report. The weight updates are dense but small in the 2-norm compared to
> the pretrained weights. The low-rank structure argument for transfer still applies
> at the level of the behavioral direction being nearly orthogonal to the dominant
> fine-tuning gradient directions — but this is an empirical claim that E8 directly tests.

**Q: What if base→IT transfer is BETTER than IT→base? What would that imply?**

> It would imply that instruction tuning sharpens the directions (increasing ‖v‖ in IT
> vs base) but doesn't change the direction itself. In that case, the IT model is a
> "better steerable" version of base: the same pretraining directions are amplified by
> alignment, making them easier to extract (larger signal-to-noise in the contrast set)
> and easier to steer with (larger projection onto the behavioral subspace).

---

## 10. Verification artifacts checklist

- [ ] Both models loaded (google/gemma-2-2b and google/gemma-2-2b-it).
- [ ] cos(v_IT, v_base) per concept at L16.
- [ ] Efficacy comparison (IT→IT vs IT→base vs base→IT) per concept × n=7 seeds.
- [ ] Behavior judge calibrated separately for each model (baseline subtracted).
- [ ] PPL + MMLU + JailbreakBench CR logged for each transfer condition.
- [ ] Result row in `EXPERIMENT_LEDGER.md`.
- [ ] Transfer concept taxonomy (pretraining-induced vs IT-induced) documented.

---

## 11. Status journal

- 2026-05-27 — hypothesis inherited from corpus E8; <20% efficacy loss threshold
  pre-registered.
- 2026-05-29 — all screening (C1–C9) used only Gemma-3-270m, Gemma-3-1B, Qwen-0.5B —
  all instruction-tuned variants only. No base-model comparison. **E8 UNTESTED.**
- 2026-05-31 — Design doc written. The Rogue Scalpel (arXiv:2509.22067) motivates
  priority for the safety concept transfer test. Requires loading base Gemma-2-2B;
  currently blocked by VRAM planning (both models = ~9 GB; manageable in 16 GB GPU).

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-A. An important untested hypothesis with high theoretical stakes.*

### Prior plausibility

**MEDIUM-to-HIGH for stylistic/factual concepts; LOW-to-MEDIUM for behavioral/alignment
concepts.** The LRH predicts pretraining-induced directions exist for concept types
well-represented in pretraining data. The Rogue Scalpel finding (arXiv:2509.22067) that
steering works on IT models is consistent with pretraining-induced directions — but the
attack only needed to steer the IT model, not transfer to the base model.

### Mechanism scrutiny

The LoRA-orthogonality argument is sound in principle but depends on the specific
alignment recipe. Gemma-2-2B-it uses SFT + DPO (not LoRA), which changes ALL weights.
Dense full-weight fine-tuning could significantly rotate the activation geometry in the
behavioral subspace, making transfer less reliable than LoRA-FT would be. This is the
primary risk for the hypothesis.

### Confounds

1. **Different vocabulary of "success"**: the LLM-judge may evaluate the IT model's
   outputs with different calibration than the base model's (which may respond
   differently to the same prompts even without steering). The doc addresses this
   with per-model baseline subtraction, which is correct.
2. **Concept leakage into contrast pairs**: if the contrast pairs used for extraction
   were generated by the IT model (and the base model would never generate such pairs),
   the extraction may be biased. Use human-authored contrast pairs to avoid this.
3. **Differential layer alignment**: the IT model's L16 may correspond to a semantically
   different layer than the base model's L16 (since fine-tuning shifts the depth of
   feature formation). A layer-scan comparison (find the base model's "best IT-vector
   injection layer") is a recommended secondary experiment.

### Does the 20% threshold specifically matter?

**Yes, for the "alignment donation" use case.** A 20% loss is the difference between
"useful as a safety intervention" and "not useful." The exact threshold should be
concept-specific: for safety/refusal, 20% loss means the base model will still refuse
80% of cases that the IT model refused under steering — potentially acceptable. For
capability-impacting concepts, the threshold may need to be tighter (10%).

### Literature precedent

No prior work directly tests IT→base steering transfer on Gemma or Llama. The closest:
Refusal direction (arXiv:2406.11717) tested their direction on the same IT model only.
The concept of "alignment donating pretraining directions" is theoretically described in
the LRH literature (Nanda et al., Elhage et al.) but not empirically tested for steering.
E8 fills a genuine gap.

### Skeptical effect-size re-prediction

IT→base efficacy loss:
- Stylistic (language) concept: < 5%, 80% CI.
- Safety/refusal concept: 10–30%, 80% CI (larger variance due to RLHF amplification).
- Behavioral/sycophancy concept: 25–60%, 80% CI (most likely to be IT-specific).

cos(v_IT, v_base): for stylistic, [0.93, 0.99]; for safety, [0.80, 0.96]; for
behavioral, [0.60, 0.90].

The 20% threshold will pass for stylistic, is uncertain for safety, and is likely to
fail for behavioral/sycophancy. A conditional finding ("transfer holds for pretraining-
strong concepts; fails for IT-specific behavioral concepts") is the most likely outcome.

### Minimum-distinguishing experiment

Two concepts (language/French + safety/refusal) × L16 × {IT→base, base→IT, IT→IT}
× n=3 (screening). Extract v_IT and v_base; measure cos(v_IT, v_base); steer each
model; record behavior delta. ~2 hours per concept. Confirm cos > 0.90 (necessary
condition) before running the behavioral evaluation.

### Verdict

**NOVEL+TESTABLE. Expected to hold for pretraining-aligned concepts; may fail for
IT-specific behaviors.** The conditional result ("transfer holds for pretraining-induced
directions, not for RLHF-shaped behaviors") is itself an important finding that would
clarify the scope of the LRH and steering research. Priority: run the stylistic concept
first (expected to work) to confirm the experimental setup, then the safety concept
(uncertain), then the sycophancy concept (likely to fail). The sequence itself is a
contribution.

---

## Pseudocode & Methodology

This hypothesis tests **IT→base transfer** of a steering vector. The knob varied is the
**target model** (IT vs base) into which a fixed IT-extracted DiffMean vector is injected;
source = DiffMean @ L16, matched relative alpha (so ‖h‖ differences between models are
neutralized).

### 1. Steering-vector recipe

```python
# Extract from the INSTRUCTION-TUNED model (METHODOLOGY §1.3); also extract a base-model
# reference vector for the cosine comparison.
H_it   = collect_activations(model_it,   tok, load_concept(behavior), layer=16)
v_it   = diffmean_vector(H_it.pos, H_it.neg)       # mean(pos)-mean(neg), from gemma-2-2b-it
H_base = collect_activations(model_base, tok, load_concept(behavior), layer=16)
v_base = diffmean_vector(H_base.pos, H_base.neg)   # from gemma-2-2b (base)
cos_it_base = cosine(v_it, v_base)                 # necessary-condition check (>=0.90)
```

### 2. Experiment procedure

```text
1. Extract v_it (IT model) and v_base (base model) at L16; record cos(v_it, v_base).
2. for target_model in {it, base}:                         # the ONE knob: which model
3.     h' = apply_operation(h, v_it/||v_it||, "relative_add", alpha_rel=0.10)
            # inject the SAME IT-extracted vector into each target
4.     beh = judge.GeminiJudge.score(generation)           # behavior efficacy
5.     PPL, CR logged.
6. efficacy_loss = 1 - beh(IT->base) / beh(IT->IT)         # baseline-subtracted per model
7. SECONDARY: base->IT transfer (inject v_base into the IT model).
8. Repeat across 3 concept types: safety/refusal, stylistic (French), behavioral (sycophancy).
```

Each model's behavior is measured as a CHANGE from its own no-steer baseline, so the
comparison is fair even when base and IT have different baseline refusal rates.

### 3. Measurement & decision rule

PRIMARY metric: IT→base **efficacy loss** at matched relative alpha. FALSIFIER (§3):
fires if IT→base efficacy loss `> 20%` on more than one concept (especially significant
for a SAFETY concept, implying the refusal direction is IT-induced not pretraining-induced).
Pre-registered (§6): loss `< 20%` (safety), `< 10%` (stylistic), `10–40%` (behavioral);
`cos(v_it, v_base) >= 0.90`; base→IT `>= 70%` of IT→IT. Verdict: SUPPORTED iff IT→base
loss `< 20%` (transfer holds ⇒ directions are pretraining-induced). Most likely outcome
is a *conditional* result: transfer holds for pretraining-strong concepts (language,
safety), fails for RLHF-shaped behaviors.

### 4. Where the code is / status

Driver: `scripts/campaign_sweep.py` loading two checkpoints (`google/gemma-2-2b` +
`google/gemma-2-2b-it` via `model.py`), reusing `extract`, `hooks.apply_operation`
(`relative_add`), and the off-family judge. Status **UNTESTED** — all screening used only
IT variants (no base model loaded); the blocker is operational (load gated base Gemma-2-2B
alongside IT, ~9 GB total in 16 GB). No new algorithmic machinery is required.

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/E8.md`.
