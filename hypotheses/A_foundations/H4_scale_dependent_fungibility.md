# H4 — Direction-Fungibility Is Scale-Dependent (the generality check on H2)

> **One-line claim:** The E7 null — that the real refusal/steering direction is
> interchangeable with norm-matched random / label-shuffled / orthogonalized
> controls under a fixed gate — WEAKENS with model scale: the real direction beats
> all controls at 9B but NOT at <= 2B, and the size of the real-minus-control gap
> CORRELATES with the model's endogenous steering-resistance (ESR) rate, turning a
> small-model curiosity into a scaling law.
>
> **Source design space:** Block A — Foundations and measurement tooling, cross-cut
> with the program "Conditional displacement, not direction" (Block G, row H4). This
> is the SCALE generalization of H2 (the mechanistic promotion of E7's null).
>
> **Implementation status:** `DESIGN + PLAN — RUN PENDING`. The control suite exists
> (`src/steering/controls.py`: `shuffled_label_vector`, `matched_norm_random`,
> `random_direction`, `extraction_stability`); the orthogonalized control and the
> ESR mid-generation refusal-recovery detector are NOT yet implemented, and the 9B
> arm CANNOT run on the 16 GB 4090 (it is CLAUDE.md's Rung-4 cross-scale check and
> requires a rented A100/L4 — the headline blocker). This doc is a pre-registered
> plan, not a result. Gated behind a hard precondition: a judge calibrated to
> AUC >= 0.85 (see §7.0).

---

## In Plain English

**What we're testing, simply:** Across the repo so far, when we steer a small model
toward refusal, the *exact* direction we push barely matters — a random push of the
same size, or a push built from scrambled labels, does almost as well as the "real"
refusal direction (this is the E7 null). H4 asks: **is that because the direction
genuinely doesn't matter, or only because the models are too small?** We predict the
real direction starts to matter on a BIG model (9B), and that the point where it
starts to matter lines up with how hard the big model fights back against being
steered (it "recovers" mid-sentence and re-refuses). If both hold, the small-model
null isn't an embarrassment — it's the small-scale end of a scaling law.

**Key terms:**
- **Steering direction:** the specific vector we add to the model's internal state to
  push it toward (or away from) refusal.
- **Control directions (the confound killers):** fake directions of the same size as
  the real one — `matched_norm_random` (random push, same magnitude),
  `shuffled_label_vector` (built from the same data with the harmful/benign labels
  scrambled), and an **orthogonalized** control (the real direction with its useful
  part projected out). If the real direction can't beat these, the *specific*
  direction isn't doing the work — the raw push is.
- **Fungible:** interchangeable. "Direction-fungibility" = you can swap the real
  direction for a control and get the same result.
- **Fixed gate:** the WHEN-to-steer decision is held constant across all directions,
  so we isolate the WHAT (the direction) from the WHEN (the gate).
- **Endogenous steering resistance (ESR):** the tendency of a (usually larger) model
  to actively resist a steering push — e.g. drift back on-task, or *recover* and
  re-refuse partway through generating — rather than being passively shoved.
- **ESR rate:** the fraction of steered generations where the model recovers /
  re-refuses mid-stream (measured by a refusal-recovery detector).
- **Scale-dependent vs scale-independent:** does the effect change as the model gets
  bigger (270M -> 1B -> 2B -> 9B), or is it the same at every size?

**Why we're doing this:** The repo's most robust finding is a null (E7/S-16/S-24: the
direction is ~97% captured by a shuffled control at 2B, NULL at 1B). A bare null is
hard to publish and easy to dismiss as "your models were too small." H4 converts it
into a *falsifiable scaling claim* and ties it to an independent phenomenon (ESR).
Either outcome is venue-grade: if the direction starts to matter at 9B and tracks
ESR, we have a scaling law; if it stays fungible even at 9B, the "direction is a
placebo" claim becomes STRONGER and more general.

**What the result would mean:** SUPPORTED ⇒ "direction-fungibility is a small-scale
artifact that dissolves once the model is big enough to resist off-direction pushes"
— a mechanistic scaling story. FALSIFIED (real still <= control at 9B, or no ESR
correlation) ⇒ the null is scale-INDEPENDENT, which makes "the direction is a
placebo" a broader, more robust claim. We win either way, and we say so honestly.

See [`../GLOSSARY.md`](../GLOSSARY.md).

---

## 1. Motivation (>= 100 words)

The repo's single most replicated result is a null about the steering DIRECTION. On
synthetic "ocean" the real DiffMean direction looked decisive (+0.135, S-15), but on
real AxBench it collapses: at 2B the real direction beats a same-norm shuffled-label
control by only +0.004 (S-16, ordinal gate FAIL, ~97% of the effect captured by the
shuffled vector), and at 1B the delta is -0.0278 with a CI that includes zero (S-24,
NULL). The 270M model is unsteerable entirely (S-23). The honest S-24 caveat is that
this null sits BELOW the detection floor of the current weak judge (AUC 0.68-0.74) at
near-floor small-model behavior — it is "not distinguishable from generic
displacement here," not "proven intrinsically generic." Two confounds therefore
co-vary in every existing run: model SCALE and judge POWER. Meanwhile the literature
on larger models reports the opposite intuition — big aligned models actively RESIST
steering, recovering mid-generation and re-asserting refusal (endogenous steering
resistance). H4 disentangles scale from judge by holding the (calibrated) judge fixed
and sweeping the model ladder 270M -> 1B -> 2B -> 9B, and predicts the real-minus-
control gap is monotone in scale and tracks the per-model ESR rate. This is exactly
CLAUDE.md's Rung-4 cross-scale check (the 9B scale arm) applied to the direction
question — the one experiment that cannot run on the 16 GB 4090 and needs a rented
GPU.

---

## 2. Formal Hypothesis (>= 50 words)

Because at small scale steered behavior sits near the coherence floor and the refusal
subspace is low-dimensional and entangled with the residual-stream norm — so ANY
same-norm displacement (random, shuffled, or orthogonalized) triggers comparable
refusal — whereas at larger scale the refusal direction is more linearly separable
and the model can resist off-direction pushes (endogenous steering resistance, ESR),
the real direction will become NON-fungible only at scale. Formal claim: under a FIXED
gate and matched off-shell displacement, the real-minus-control delta on
safety-under-attack at matched coherence is <= 0 with a bootstrap 95% CI overlapping
zero at 270M, 1B, and 2B, but > 0 with a CI strictly above zero at 9B; and across the
four-rung ladder the real-minus-control delta is positively rank-correlated
(Spearman) with the model's ESR multi-attempt rate (the fraction of steered
generations exhibiting mid-generation refusal recovery). The mechanism is increasing
linear separability + resistance, not judge power, which is held fixed at AUC >= 0.85.

---

## 3. Falsifier (>= 30 words)

FALSIFIED if the real-minus-control delta is still <= 0 (bootstrap 95% CI includes 0)
at 9B under the calibrated judge, OR if the delta does NOT positively correlate with
the per-model ESR rate across the 270M/1B/2B/9B ladder (Spearman rho <= 0 or CI
spanning 0). Note honestly: a 9B null does NOT vindicate steering — it makes the
"direction is a placebo" claim STRONGER and scale-INDEPENDENT (fungibility holds at
every tested scale), so the null is robust either way; H4 distinguishes the
*scaling-law* story from the *universal-null* story, and only the scaling-law story
is what "SUPPORTED" means here. A confound that must be ruled out before declaring
SUPPORTED: a 9B real-minus-control gap that vanishes when displacement and coherence
are matched (i.e. the gap is a norm/coherence artifact, not a direction effect).

---

## 4. Citations (Citation Rigor >= 80 words)

```
Arditi, Andy, et al. 2024 NeurIPS 'Refusal in Language Models Is Mediated by a
Single Direction' (arXiv:2406.11717) — the refusal direction whose fungibility H4
scale-checks; "single direction" predicts HIGH separability, which §5 argues
emerges with scale and is what would make the real direction non-fungible at 9B.

Panickssery, Aryan, et al. 2023 arXiv 'Steering Llama 2 via Contrastive Activation
Addition' (arXiv:2312.06681) — CAA reports that larger models need different (often
SMALLER, relative) alpha and steer differently; the scale-dependence of the steering
response is the empirical seed of H4's prediction.

Korznikov, A., et al. 2026 ICML 'The Rogue Scalpel: Activation Steering Compromises
LLM Safety' (arXiv:2509.22067) — finding that off-manifold displacement (any same-
norm push) degrades behavior independent of direction; this is precisely the
small-scale fungibility mechanism (§5), and its scale-dependence is H4's question.

Templeton, A., et al. 2024 Anthropic 'Scaling Monosemanticity: Extracting
Interpretable Features from Claude 3 Sonnet' [UNVERIFIED arXiv id] — feature
separability and linear structure sharpen with scale; supports the §5 claim that the
refusal subspace becomes more linearly isolable (hence direction-specific) at 9B.

Wei, Jason, et al. 2022 TMLR 'Emergent Abilities of Large Language Models'
(arXiv:2206.07682) — the general scale-emergence frame H4 instantiates: a quantity
that is null at small scale and non-null past a threshold size.

[UNVERIFIED] 'Endogenous Steering Resistance' / mid-generation refusal-recovery
under activation steering in large aligned models (e.g. Llama-3.3-70B) — the exact
arXiv id is NOT confirmed; cited as [UNVERIFIED] pending corpus verification. H4's
ESR metric and the correlation claim depend on this phenomenon; if no primary source
is confirmed, the ESR rate is measured DIRECTLY here (a refusal-recovery detector)
and the citation is dropped rather than asserted.
```

---

## 5. Mechanism

### 5.1 Why the direction is fungible at small scale

At 270M-2B, two facts compound. (i) **Coherence floor:** steered behavior peaks at
the gentlest push and collapses by alpha~0.2 (S-23, S-24, S-17), so the usable
displacement is tiny and the model is operating near the edge of coherence — any
same-norm shove (real, random, shuffled) pushes it toward the same degraded /
refusal-adjacent region. (ii) **Norm entanglement + low-dim refusal subspace:** the
refusal behavior is dominated by the radial component (Δ‖h‖), per the Rogue Scalpel
off-manifold mechanism and the CRH; a small model's refusal subspace is low-rank and
not cleanly separable, so `matched_norm_random`, `shuffled_label_vector`, and an
orthogonalized control all land "close enough." Result: real-minus-control ~ 0
(the E7 null), which is what S-16/S-24 measure.

### 5.2 Why the direction should matter at scale

A larger model has (a) a more linearly separable, higher-effective-rank refusal
representation (the "single direction" of Arditi is cleaner the bigger the model),
and (b) **endogenous steering resistance**: it can re-assert refusal mid-generation,
so an OFF-direction push (random/shuffled/orthogonalized) is actively corrected back
while the ON-direction push aligns with the model's own refusal machinery and is not
resisted. The consequence is a real-minus-control gap that OPENS with scale, and that
opens *in proportion to* how much the model resists — i.e. the gap tracks the ESR
rate. This converts the null into a scaling law with a mechanistic, measurable
mediator (ESR), not just a size axis.

### 5.3 The controls (reused, score-agnostic)

All from `src/steering/controls.py`, slotting into the existing pipeline in place of
the real DiffMean vector at the SAME off-shell displacement:
- `matched_norm_random(v_real, seed)` — random direction at exactly ‖v_real‖.
- `shuffled_label_vector(pos, neg, seed)` — DiffMean after a random label re-partition.
- **orthogonalized** — v_real with its top-k refusal-subspace component projected out
  (the one new control to implement; NOT yet in controls.py), rescaled to ‖v_real‖.
- `extraction_stability(pos, neg)` — gate that v_real itself is stable
  (mean cosine-to-full near 1.0) BEFORE any cross-scale comparison is trusted.

### 5.4 The ESR mediator (the new measurement)

ESR rate = fraction of steered generations in which a **mid-generation refusal-
recovery detector** fires: the model begins to comply (or drift on-task) under
steering and then re-asserts a refusal / re-aligns before the end of the generation.
Implemented as a sliding-window refusal-marker + judge pass over generation prefixes
vs full generations. ESR is measured per model on the SAME steered prompts used for
the direction test, so the correlation is within-condition.

---

## 6. Predicted Delta

| Metric | Predicted | Rationale |
|---|---|---|
| real − control delta @ 270M (safety-under-attack, matched coherence) | [-0.05, +0.01], CI ∋ 0 | unsteerable floor (S-23); fungible |
| real − control delta @ 1B | [-0.06, +0.01], CI ∋ 0 | replicates S-24 NULL (-0.028) |
| real − control delta @ 2B | [-0.01, +0.02], CI ∋ 0 | replicates S-16 (+0.004, ordinal FAIL) |
| real − control delta @ 9B | [+0.03, +0.15], CI > 0 | core claim: non-fungible at scale |
| ESR rate (mid-gen recovery) @ 270M / 1B / 2B / 9B | ~[0.00, 0.03, 0.06, 0.20] | resistance rises with scale |
| Spearman(real−control delta, ESR rate) over 4 rungs | > 0, CI > 0 | the mediator correlation (falsifier) |
| 9B gap survives displacement+coherence matching | True | else the gap is a norm artifact (§3 confound) |

Pre-registered; [NEEDS VERIFICATION]. All numbers are predictions, not results. The
4-point Spearman has very low power (n=4 models); the correlation is REPORTED with its
exact CI and treated as suggestive, with per-rung deltas as the primary evidence.

---

## 7. Experimental Protocol

### 7.0 HARD PRECONDITION (gate before any H4 run)

**The judge must be calibrated to AUC >= 0.85** against ground-truth refuse/comply
labels (PREREGISTRATION §6 + Amendment 1; the same bar M-series uses). The current
behavior judge is AUC 0.68 and the Qwen safety judge is conservative — both below
bar. H4 SPECIFICALLY exists to separate scale from judge power, so running it under a
weak judge would re-introduce the exact confound it is designed to remove. Until the
judge clears AUC >= 0.85, H4 runs are SCREENING ONLY and cannot ground the
real-minus-control delta. Pre-check (cheap, judge-free): `extraction_stability` on
v_real at every scale (abort a scale arm if mean cosine-to-full < ~0.9).

### 7.1 The blocker (state prominently)

**The 9B arm CANNOT run on the 16 GB 4090.** gemma-2-9b-it at 4-bit needs ~6-7 GB for
weights plus activation-extraction overhead across many prompts and four control
directions; with the ESR mid-generation detector running full JBB-under-attack +
XSTest it does not fit the inner-loop budget. The 9B arm REQUIRES a rented A100/L4
(CLAUDE.md Rung-4 cross-scale check). The 270M/1B/2B arms run on the 4090 and can be
completed first (and largely REPLICATE S-23/S-24/S-16, lowering risk); the 9B arm is
the single decisive, blocked experiment. No external claim until the 9B arm runs.

### 7.2 Primary experiment

- **Models (the ladder):** gemma-3-270m-it (floor anchor, known unsteerable, S-23),
  gemma-3-1b-it, gemma-2-2b-it, gemma-2-9b-it (RENTED GPU).
- **Direction:** refusal direction from real harmful (JBB-harmful) vs benign
  (JBB-benign / XSTest-safe) PROMPTS — input-triggered, NOT AxBench concepts (the
  S-25 lesson: conditioning's clean testbed is input-triggered safety steering).
- **Controls:** `matched_norm_random`, `shuffled_label_vector`, orthogonalized — all
  at the SAME off-shell displacement as v_real (matched coherence, matched Δ‖h‖).
- **Endpoint:** safety-under-attack ASR (JBB after `adversarial.py` transforms so the
  unsteered baseline ASR is non-zero, per PREREG Amendment 1) at matched coherence;
  report real-minus-control per model.
- **Gate:** FIXED across all directions (the H2 device) so only the direction varies.
- **ESR:** mid-generation refusal-recovery detector run on the same steered prompts.
- **Seeds / units:** the replication unit is **resampled extraction pairs × prompt
  bootstrap** (PREREG Amendment 1c — greedy decoding has ~0 variance; NOT decode
  seeds), n >= 7.
- **Stats:** paired Wilcoxon (real vs each control), bootstrap 95% CI on each
  real-minus-control delta, Holm-Bonferroni across the {270M, 1B, 2B, 9B} × {3
  controls} family, Spearman(delta, ESR) across the 4 rungs with its CI.

### 7.3 Where it shines

The contrast is sharpest at the LADDER ENDS: 270M (fungible, floor) vs 9B
(non-fungible, resistant). The 4090 arms establish the null end cheaply; the rented-
GPU 9B arm is where SUPPORTED/FALSIFIED is decided. Report the 270M/1B/2B null FIRST
(reproduces prior findings, builds trust), the 9B contrast SECOND.

---

## 8. Cross-References

- **E7** ([`E7_norm_relative_alpha.md`](E7_norm_relative_alpha.md)) — the null this
  generalizes (S-15 synthetic win → S-16/S-24 real null); H4 is E7's direction
  question swept across scale.
- **H2** (`H2_direction_fungible_under_gate.md`, written in parallel) — the claim this
  scale-checks: H2 promotes the E7 null to "direction is fungible under a fixed gate";
  H4 asks whether that holds at 9B.
- **PROGRAM** (`../PROGRAM_conditional_displacement.md`) — "Conditional displacement,
  not direction"; H4 is the generality check (H4 of the program).
- **`src/steering/controls.py`** — `shuffled_label_vector`, `matched_norm_random`,
  `random_direction`, `extraction_stability`; the orthogonalized control + ESR
  detector are the new code H4 adds.
- **S-16 / S-24 / S-23** (FINDINGS) — the 2B/1B/270M null arms H4 reproduces.
- **M8 / E15 / S-26** — the gate is the contribution if the direction stays fungible;
  H4 tells us whether that conclusion is scale-bounded.
- **PREREGISTRATION.md** Amendment 1 (safety-under-attack endpoint, judge AUC >= 0.85,
  bootstrap-unit redefinition); **IDEA_TABLE.md** Block G row H4.

---

## 9. Committee Q&A

**Q: A 4-point Spearman correlation across only 4 models is statistically empty.**
> Agreed, and we say so: the per-rung real-minus-control deltas are the PRIMARY
> evidence; the ESR correlation is a low-power mediator check reported with its exact
> CI as suggestive, not decisive. The scaling claim survives or falls on the 9B delta
> being CI-clean while the smaller rungs are CI-zero — a categorical, not a
> correlational, result.

**Q: If 9B is also null, haven't you just failed?**
> No — a 9B null FALSIFIES the scaling-law story but STRENGTHENS the "direction is a
> placebo" story, making it scale-independent and far more general (the null holds
> everywhere we can test). We pre-register both readings and report whichever the data
> gives; only the scaling-law reading is labeled "SUPPORTED," but the universal-null
> reading is an equally publishable, arguably stronger, negative result.

**Q: How do you know a 9B gap is the DIRECTION and not just that 9B tolerates more
displacement?**
> By matching off-shell displacement (Δ‖h‖) and coherence between real and control
> directions at every scale (the §3 confound check). If the gap vanishes once
> displacement and coherence are matched, we report it as a norm artifact, not a
> direction effect — SUPPORTED requires the gap to survive matching.

**Q: Why not just run 9B on the 4090 at lower precision?**
> The binding cost is not just weights — it is four control directions × the
> adversarial JBB + XSTest sweep × the ESR mid-generation detector (which judges
> generation prefixes). That throughput does not fit the inner loop on 16 GB. This is
> exactly the Rung-4 cross-scale check CLAUDE.md reserves for rented GPU.

---

## 10. Verification Checklist

- [ ] Judge calibrated to AUC >= 0.85 (PRECONDITION) OR result labeled SCREENING.
- [ ] `extraction_stability` on v_real >= ~0.9 mean cosine-to-full at every scale.
- [ ] Refusal direction extracted from real harmful-vs-benign PROMPTS (not AxBench).
- [ ] All three controls (matched-norm random, shuffled-label, orthogonalized) at
      MATCHED off-shell displacement + matched coherence.
- [ ] Safety-under-attack endpoint (JBB after `adversarial.py`) with non-zero
      baseline ASR (PREREG Amendment 1).
- [ ] real − control delta + bootstrap 95% CI reported per model (270M/1B/2B/9B).
- [ ] 9B gap re-checked after displacement+coherence matching (the §3 confound).
- [ ] ESR mid-generation refusal-recovery detector implemented + per-model ESR rate.
- [ ] Spearman(delta, ESR) over 4 rungs reported WITH its CI (flagged low-power, n=4).
- [ ] Holm-Bonferroni across {scale × control} family; bootstrap unit = extraction-
      pair × prompt bootstrap, n >= 7 (NOT decode seeds).
- [ ] 9B arm run on rented A100/L4 (documented); IDEA_TABLE Block G row H4 + PROVENANCE/
      H4.md created on first run.

---

## 11. Status Journal

- 2026-06-10 — Design doc created as H4 of the program "Conditional displacement, not
  direction" (the generality / scale check on H2). Status: **DESIGN + PLAN — RUN
  PENDING, BLOCKED ON RENTED GPU.** The 270M/1B/2B arms reproduce existing screening
  (S-23 270M-unsteerable, S-24 1B-NULL, S-16 2B-+0.004); the decisive 9B arm cannot
  run on the 16 GB 4090 and requires a rented A100/L4 (CLAUDE.md Rung-4 cross-scale
  check). Blocked on: (a) judge calibration to AUC >= 0.85 (the §7.0 precondition that
  separates scale from judge power — the entire point of H4); (b) the orthogonalized
  control (new code, projecting out the refusal subspace, rescaled to ‖v_real‖);
  (c) the ESR mid-generation refusal-recovery detector (new code); (d) rented-GPU
  access for the 9B arm. Pre-registers BOTH outcomes honestly: SUPPORTED ⇒ scaling
  law; FALSIFIED ⇒ scale-independent null (a stronger, equally publishable negative).
  No results. The existing `controls.py` suite (`shuffled_label_vector`,
  `matched_norm_random`, `extraction_stability`) is reused unchanged.

---

## Pseudocode & Methodology

```python
# H4: is the E7 direction-null scale-dependent? Reuses controls.py at matched
# displacement; sweeps the model ladder; correlates the gap with ESR.
import numpy as np
from steering.controls import (matched_norm_random, shuffled_label_vector,
                               extraction_stability)

LADDER = ["gemma-3-270m-it", "gemma-3-1b-it", "gemma-2-2b-it",
          "gemma-2-9b-it"]   # last arm => RENTED A100/L4, cannot run on 4090

def real_minus_control(model, pos, neg, *, judge, gate, n_boot=2000):
    assert judge.auc >= 0.85, "PRECONDITION: judge AUC >= 0.85 (else SCREENING only)"
    v_real = diffmean_vector(pos, neg)
    assert extraction_stability(pos, neg)["mean_cosine_to_full"] >= 0.9
    controls = {
        "matched_norm_random": matched_norm_random(v_real, seed=0),
        "shuffled_label":      shuffled_label_vector(pos, neg, seed=0),
        "orthogonalized":      orthogonalize(v_real, refusal_subspace),  # TO WRITE
    }
    # SAME fixed gate + SAME off-shell displacement for every direction (the H2 device)
    s_real = safety_under_attack(model, v_real, gate=gate)        # matched coherence
    deltas = {k: s_real - safety_under_attack(model, vc, gate=gate)
              for k, vc in controls.items()}
    return deltas, esr_rate(model, v_real, gate=gate)            # ESR mediator

# PRIMARY: per-model bootstrap CI on each real-minus-control delta.
#   predict CI ∋ 0 at 270m/1b/2b ; CI > 0 at 9b.
# MEDIATOR: Spearman(delta, ESR) across the 4 rungs (low power, n=4, report CI).
# CONFOUND CHECK: re-run 9b with displacement+coherence matched; gap must survive.
```

Decision rule: SUPPORTED iff (real − control) CI ∋ 0 at <= 2B AND CI > 0 at 9B AND the
9B gap survives displacement/coherence matching. FALSIFIED iff 9B CI ∋ 0 OR
Spearman(delta, ESR) <= 0 — reported as the (stronger) scale-INDEPENDENT null. See
[`../METHODOLOGY.md`](../METHODOLOGY.md).

---

## Provenance & Tracing

`DESIGN + PLAN — no experiments yet.` On first run, create `PROVENANCE/H4.md` with the
exp#, reproduce command, and artifact links. The 270M/1B/2B arms run on the 4090; the
9B arm runs on a rented A100/L4 (documented separately). Reproduce (once the judge
precondition, the orthogonalized control, and the ESR detector land):

```bash
# PYTHONPATH=src python scripts/run_scale_fungibility.py \
#   --models gemma-3-270m-it gemma-3-1b-it gemma-2-2b-it gemma-2-9b-it \
#   --judge-model Qwen/Qwen2.5-7B-Instruct --controls matched_norm_random shuffled_label orthogonalized \
#   --endpoint safety_under_attack --measure-esr --no-log   # 9b arm => rented GPU; TO BE WRITTEN
```
