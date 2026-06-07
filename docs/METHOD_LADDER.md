# Method Ladder — Conditional Safety Steering

Each rung must be cleared before spending compute on the next. Every rung is
defined by its gate: a concrete, pre-registered criterion that must pass on
the specified benchmark before promotion.

---

## Rung 0: Refusal direction extracted and validated

**What:** Extract an Arditi-style refusal / safe-completion direction from real
safety-contrast activations (harmful prompts vs safe-completion responses).

**Gate:** cos(refusal_direction, DiffMean_baseline) > 0.85 on >= 50 contrast
pairs; direction is stable across bootstrap resamples (bootstrap cos to
full-data direction >= 0.90).

**Instrument:** activation extraction only (no generation required).

**Current status:** NOT STARTED. The cast.py file exists but the refusal
direction is not yet extracted from real safety contrasts.

---

## Rung 1: In-forward conditional gate wired (CAST-style)

**What:** A real in-forward conditional pipeline: read h@L_c -> live gate
decision -> masked steering write @L_b, all in ONE forward pass. Not offline
numpy patched in after the fact.

**Gate:** On 10 held-out harmful prompts and 10 benign prompts: gate fires
on >= 9/10 harmful, fires on <= 1/10 benign (false-positive rate <= 10%).
Latency overhead vs no-gate baseline: <= 20%.

**Instrument:** rule-based harm classifier on the 20 prompts (sufficient for
Rung 1; a calibrated classifier is required by Rung 2).

**Current status:** NOT STARTED. gate.py is offline numpy; hooks.py not wired.

---

## Rung 2: Multi-intent steering defined and implemented

**What:** Compose K >= 2 safety directions (one per harm category), applied
independently per detected intent in a single forward pass. Gram-Schmidt
orthogonalization (E19) applied to the K directions before composition.

**Gate:** On a held-out multi-intent eval set (K >= 2 simultaneous harm
categories): per-category refusal rate >= 80% at XSTest over-refusal <= 5%.
Composite formula applied across all five axes; safety axis uses a calibrated
classifier (Llama-Guard-3 or ShieldGemma) with >= 0.90 human agreement on
>= 100 labeled items.

**Instrument:** calibrated safety classifier + XSTest + real MMLU (>= 500
items, in-run).

**Current status:** NOT STARTED.

---

## Rung 3: Over-refusal control operationalized

**What:** Demonstrate that the conditional gate (vs unconditional steering)
reduces XSTest over-refusal by >= 70% at matched or better harmful-refusal
rate. Iso-behavior capability curve (E46) traced to confirm behavior and
capability are separately tunable.

**Gate:** XSTest over-refusal rate: gated <= 1%, unconditional > 5% at
matched behavior. MMLU drop vs unsteered baseline: <= 2 pp. Both conditions
confirmed at n >= 7 seeds with paired Wilcoxon p < 0.05.

**Instrument:** JailbreakBench ASR (primary attack metric) + StrongREJECT
rubric grader + XSTest + MMLU >= 500.

**Current status:** NOT STARTED.

---

## Rung 4: Adversarial evaluation (SOTA claim gate)

**What:** The method must withstand adaptive attacks. Reproduce the
Rogue-Scalpel 20-vector universal attack as a red-team probe; it must be
neutralized. Test GCG suffixes, PAIR, and AutoDAN. Implement and ablate the
five-layer guard (A: refusal-formation subspace lock; B: norm/manifold
clamp; C: avoid fragile mid-layers; D: dual-forward verdict check; E:
conditional gate).

**Gate:** ASR under adaptive attack: <= 5% (JailbreakBench + HarmBench).
Universal attack neutralized (attack success drops to near baseline).
XSTest over-refusal: <= 1%. MMLU drop: <= 2 pp. Pareto-dominates CAST and
prompting baseline on the ASR vs over-refusal frontier.

**Instrument:** JailbreakBench + StrongREJECT + HarmBench + XSTest + MMLU +
human red-team.

**Current status:** NOT STARTED.

---

## Rung 5: Publication-grade (SOTA claim)

**What:** Full multi-axis win with ablations, scale check on Gemma-2-9B-it,
cross-family demonstration (>= 2 model families), and independent external
review.

**Gate:** All Rung 4 gates hold across >= 2 model families and two scales.
Pre-registered success criterion confirmed. Independent (different-family)
external review issued.

**Current status:** NOT STARTED.

---

## Sequencing note

The outcome and success criterion in README.md defines the Rung 3 gate as
the minimum bar for any publication claim. Rungs 0-2 are P0 infrastructure
work (see audits/reviews/IMPROVEMENTS_100.md, groups A and B). Do not run
expensive benchmarks (Rung 3+) until the method passes Rung 1 and Rung 2
gates — a gate failure at a cheap rung is a cheaper failure.
