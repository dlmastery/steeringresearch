# H2 — The Steering Direction Is Fungible Under the Gate

> **One-line claim:** With the conditional gate fixed, replacing the real refusal/
> steering direction with a norm-matched **random**, **label-shuffled**, or
> **orthogonalized** direction does **NOT** degrade safety-under-attack at matched
> coherence. The gate's fire-rate moves the safety curve; the direction does not.
>
> **Source design space:** Program "Certified Conditional Displacement" (H2 — the
> bold null). Promotes the E7 / S-16 / S-24 null from "disappointing" to a
> mechanistic claim. See `hypotheses/PROGRAM_conditional_displacement.md`.
>
> **Implementation status:** `DESIGN + PLAN — RUN PENDING`. Controls already exist
> (`src/steering/controls.py`: shuffled_label_vector, matched_norm_random); the
> gate exists (`gate.CosineGate`, M8). The decisive, CHEAPEST experiment in the
> program — run it FIRST to learn which paper this is. Hard precondition: judge
> calibrated to AUC >= 0.85 (H0).

---

## In Plain English

**What we're testing, simply:** Most steering research obsesses over finding *the
right direction* to push the model's internal state. Our own data already hints the
direction barely matters at small scale — a scrambled fake vector got ~97% of the
effect of the real one (E7). This asks the blunt question: **if we keep the "when to
push" gate fixed, does it matter AT ALL which direction we push?** If three different
fake directions work as well as the real one, then the real contribution is the gate
and the shove — not the meaning of the vector.

**Key terms:**
- **Steering direction:** the vector added to the model's internal state.
- **Refusal/real direction:** the principled direction (DiffMean / Arditi) meant to
  make the model refuse.
- **Controls:** fake directions matched in size (norm) but stripped of meaning —
  **random** (a random unit vector), **label-shuffled** (built from the same data
  with the harmful/harmless labels scrambled), **orthogonalized** (a vector forced to
  be perpendicular to the real one).
- **The gate:** the rule that decides WHEN to steer (only on harmful-looking inputs).
- **Safety-under-attack:** harmful-compliance rate when the prompts are adversarial
  (the only regime with headroom; aligned models refuse everything otherwise, S-22).
- **Matched coherence:** compared at equal output quality, so we're not just trading
  safety for gibberish.

**Why we're doing this:** It is the single most decisive, cheapest experiment in the
program. If the direction is fungible, the whole field's main activity (direction-
finding) is the wrong lever at this scale, and our contribution is gating + budget.
If the real direction wins, we are back to a (still-fine) direction paper. Either
answer tells us which paper to write — so it runs FIRST.

**What the result would mean:** Fungibility SUPPORTED ⇒ a genuinely surprising,
citable result ("direction is a placebo under the gate at <=2B"). FALSIFIED ⇒ the
direction carries real signal the gate can exploit; pivot to characterizing it.

See [`../GLOSSARY.md`](../GLOSSARY.md).

---

## 1. Motivation (>= 100 words)

The repo's central screening result is a null that has been treated as an
embarrassment: on real AxBench concepts the DiffMean direction beats a norm-matched
shuffled-label control by only +0.004 at 2B (~97% of the effect captured by the
control) and is NULL at 1B (S-16, S-24). The honest caveat is that this may sit below
the judge's detection floor (AUC 0.68-0.74) — which is precisely why H0 (judge
calibration to >=0.85) gates this test. But if the null survives a trustworthy judge,
it is not an embarrassment; it is a *finding*: at small scale the steering effect is
carried by the displacement magnitude and the decision of WHEN to apply it, not by
the semantic content of the vector. AxBench itself earned an ICML spotlight by showing
SAEs lose to simple baselines — careful negatives with airtight controls build
reputation faster than weak positives. H2 promotes the null by adding two controls the
field rarely runs together (random + orthogonalized, alongside shuffled) and by
testing them WHERE IT MATTERS: under the gate, on safety-under-attack with real
headroom, at matched coherence.

---

## 2. Formal Hypothesis (>= 50 words)

Because at <=2B the refusal subspace is low-dimensional and entangled with the
activation norm (so any same-norm displacement at a harm-relevant token nudges the
model similarly), and because the gate — not the vector — decides whether a
displacement is applied at all, fixing the gate and substituting a norm-matched
random, label-shuffled, or orthogonalized direction for the real refusal direction
will yield safety-under-attack (harmful-compliance reduction) curves that overlap the
real direction's curve within a 95% bootstrap CI at matched coherence, with only the
gate fire-rate shifting the curve. Formal claim: for each control c in {random,
shuffled, orthogonal}, |ASR_real(alpha) - ASR_c(alpha)| has a 95% bootstrap CI
including 0 at every alpha on the matched-coherence frontier, at 1B AND 2B, under a
judge calibrated to AUC >= 0.85.

---

## 3. Falsifier (>= 30 words)

FALSIFIED if the real direction reduces safety-under-attack by a CI-clean margin over
ALL THREE controls at matched coherence (i.e. the direction carries exploitable
signal the controls lack) at 1B or 2B. Partial: if real beats random+orthogonal but
ties shuffled, the effect is "any harm-correlated direction", not "the" direction —
report as PARTIAL-FUNGIBLE. A null result under an UNCALIBRATED judge (AUC < 0.85)
does NOT count — it is uninterpretable (H0 precondition).

---

## 4. Citations (Citation Rigor >= 80 words)

```
Arditi, Andy, et al. 2024 arXiv 'Refusal in Language Models Is Mediated by a
Single Direction' (arXiv:2406.11717) — the "single direction" claim H2 stress-tests:
if a norm-matched random vector under the gate matches it, refusal mediation at
<=2B is a norm/gate effect, not a specific-direction effect.

Wu, Zhengxuan, et al. 2025 ICML 'AxBench: Steering LLMs? Even Simple Baselines
Outperform Sparse Autoencoders' (arXiv:2501.17148) — the benchmark + the precedent
that a rigorous NEGATIVE (SAEs lose to baselines) is a high-value contribution; H2
is the same move for direction-specificity.

Turner, Alexander Matt, et al. 2024 arXiv 'Activation Addition: Steering Language
Models Without Optimization' (arXiv:2308.10248) — ActAdd treats the difference
vector as semantically meaningful; H2's controls isolate whether the meaning or the
displacement does the work.

Panickssery, Nina, et al. 2024 arXiv 'Steering Llama 2 via Contrastive Activation
Addition' (arXiv:2312.06681) — CAA's contrastive direction; the shuffled-label
control is the matched null for a contrastive vector.
```

---

## 5. Mechanism

### 5.1 The controls (already in `controls.py`)

- **random:** a fixed random unit vector, scaled to the real direction's norm.
- **shuffled:** `shuffled_label_vector` — DiffMean over the same activations with the
  harmful/harmless labels permuted (same data, no real contrast).
- **orthogonal:** `v_orth = unit(v_rand - (v_rand . v_hat) v_hat)` — a random
  direction projected off the real one (NEW small helper).
All applied via `relative_add` at matched alpha so the per-token displacement
`alpha * ||h|| * v_hat` has identical magnitude across arms.

### 5.2 Why fungibility is plausible at small scale

At <=2B the "refusal" response is triggered by crossing a low-dimensional region; a
same-norm push at a harm-relevant token (selected by the gate) perturbs the residual
enough to tip behavior regardless of the push's precise orientation. The gate, not the
vector, supplies the selectivity. (H7 predicts this breaks at 9B where the refusal
direction is more separable and the model can resist off-direction pushes.)

### 5.3 The gate is held fixed

The SAME cosine gate (real-direction-derived threshold, M8/H1) decides firing for
ALL arms — so only the WRITE direction changes, isolating the direction's
contribution. (A second variant also swaps the gate's read direction, to test whether
the gate itself needs the real direction — reported separately.)

---

## 6. Predicted Delta

| Metric | Predicted | Rationale |
|---|---|---|
| ASR_real - ASR_shuffled (matched coh, 1-2B) | CI includes 0 | E7/S-16: ~97% captured |
| ASR_real - ASR_random (matched coh) | CI includes 0 | core fungibility claim |
| ASR_real - ASR_orthogonal (matched coh) | CI includes 0 | strongest control |
| Coherence (all arms, matched alpha) | equal within CI | same displacement norm |
| Gate fire-rate effect on ASR | large, monotone | the real lever |
| At 9B (H7 preview) | real - controls > 0 | scale-emergence |

Pre-registered; [NEEDS VERIFICATION]. Predictions, not results.

---

## 7. Experimental Protocol

### 7.0 HARD PRECONDITION
Judge calibrated to AUC >= 0.85 (H0). A null under a weaker judge is uninterpretable.
Headroom: run under the adversarial harness (PAIR/prefill) OR on the abliterated 1b
(~100% baseline ASR) so ASR is not floored.

### 7.1 Primary experiment
- **Models:** gemma-3-1b-it + gemma-2-2b-it (+ abliterated 1b for headroom).
- **Arms:** {real, random, shuffled, orthogonal} x alpha-grid, gate FIXED (real-derived).
- **Eval:** JBB-100 under attack (ASR) + XSTest-250 (over-refusal) + WikiText PPL (coherence).
- **Match:** compare arms at matched coherence (interpolate to equal PPL), not matched alpha.
- **Stats:** bootstrap over prompts + extraction-pair resamples (NOT decode seeds; greedy
  ~0 variance), paired Wilcoxon real-vs-each-control, Holm-Bonferroni across the 3 controls
  x 2 scales, 95% bootstrap CI on each delta.

### 7.2 Where it's decisive
If all three controls overlap the real arm within CI at matched coherence at 1B AND
2B → fungibility SUPPORTED → the program's contribution is gating + budget (H1/H4), and
H7 becomes the headline scale test. Run this BEFORE H1 so the program knows its thesis.

---

## 8. Cross-References

- **E7** (`A_foundations/E7_*`) — the null this promotes.
- **H1 / M8** (`B_conditional/M8_conformal_gate.md`) — the fixed gate.
- **H7** (`A_foundations/H4_scale_dependent_fungibility.md`) — the scale check on this claim.
- **H4** (`D_geometry/H3_displacement_budget.md`) — "how far" complements "which direction".
- **controls.py**, **PROGRAM_conditional_displacement.md**, IDEA_TABLE Block G row H2.

---

## 9. Committee Q&A

**Q: Isn't "the null survives" just a weak/insensitive experiment?**
> Guarded by H0 (judge >=0.85) and by powering over full JBB/XSTest with prompt
> bootstrap. A null with airtight controls and adequate power is a finding, not an
> absence of one — the AxBench precedent is explicit.

**Q: Random/orthogonal vectors should produce gibberish, not safety.**
> At matched coherence we control exactly for that: we compare arms at equal PPL. If a
> random direction only "works" by degrading coherence, it is NOT fungible and H2 is
> falsified. The claim is fungibility AT MATCHED COHERENCE.

**Q: Why run this before the conformal gate (H1)?**
> Because its outcome decides which paper H1 lives in: "the gate makes generic
> displacement safe" (fungible) vs "the gate plus the right direction" (not). Cheapest,
> most informative — classic decisive-experiment-first.

---

## 10. Verification Checklist

- [ ] Judge AUC >= 0.85 (H0) OR result labeled uninterpretable.
- [ ] Headroom present (under-attack ASR or abliterated baseline > ~0.3).
- [ ] All 4 arms at IDENTICAL per-token displacement norm (matched alpha grid).
- [ ] Compared at MATCHED COHERENCE (PPL-interpolated), not matched alpha.
- [ ] Three controls: random, shuffled, orthogonal (orthogonal helper added to controls.py).
- [ ] Gate held fixed (real-derived) across arms; gate-swap variant reported separately.
- [ ] Bootstrap CI on each real-vs-control delta; Holm across controls x scales.
- [ ] IDEA_TABLE Block G row H2 updated; PROVENANCE/H2.md on first run.

---

## 11. Status Journal

- 2026-06-10 — Design doc created (the decisive cheap test of the program; "run H2
  first"). Status: DESIGN + PLAN, RUN PENDING. Blocked on: (a) H0 judge calibration
  >= 0.85; (b) headroom (adversarial harness wiring OR the abliterated model, already
  downloaded). Controls (random, shuffled) exist in controls.py; orthogonal helper is
  a trivial add. The E7 null (S-16/S-24) is the screening precursor this formalizes.

---

## Pseudocode & Methodology

```python
# H2: is the direction fungible under a FIXED gate? (controls.py + gate.CosineGate)
from steering.controls import shuffled_label_vector, matched_norm_random
import numpy as np

def orthogonal_control(v_hat, seed):                       # NEW trivial helper
    r = matched_norm_random(v_hat, seed)
    r = r - (r @ v_hat) * v_hat
    return r / (np.linalg.norm(r) + 1e-8)

# arms share the SAME gate (real-derived threshold) and the SAME alpha grid:
arms = {"real": v_real, "random": matched_norm_random(v_real, 0),
        "shuffled": shuffled_label_vector(pos, neg, seed=1),
        "orthogonal": orthogonal_control(v_real, 2)}
# for each arm: gate-fire on harmful-under-attack -> steer @alpha -> judge ASR; judge over-refusal;
# measure PPL; then COMPARE arms at matched PPL. FUNGIBLE iff every real-vs-control CI includes 0.
```

Decision rule: PRIMARY = real-vs-control ASR delta at matched coherence; FUNGIBLE
(SUPPORTED) iff all three CIs include 0 at 1B and 2B under a >=0.85 judge; FALSIFIED if
real CI-clean beats all three. See [`../METHODOLOGY.md`](../METHODOLOGY.md).

---

## Provenance & Tracing

`DESIGN + PLAN — no experiments yet.` On first run create `PROVENANCE/H2.md`. Reproduce
(once H0 + headroom land): a committed `scripts/run_h2_fungibility.py` (TO BE WRITTEN)
sweeping {real,random,shuffled,orthogonal} x alpha under the fixed gate on the
abliterated 1b + 2b-under-attack, judged by the calibrated Qwen judge.
