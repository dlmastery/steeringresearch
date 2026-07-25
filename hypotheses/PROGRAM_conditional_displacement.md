# Research Program — Certified Conditional Displacement

> **The one-line thesis:** At small scale, activation steering is **generic
> displacement that you gate**. The steering *direction* barely matters (E7: a
> scrambled control captures ~97% of the effect); what matters is **WHEN** you push
> (the gate — the only thing with positive signal, detector AUC ~0.75, +0.125
> off-target fluency saved), **HOW FAR** you push before coherence breaks (N17:
> off-manifold displacement predicts incoherence, rho=+0.585), and **at WHAT SCALE**
> the direction starts to matter at all. The contribution is therefore not a better
> vector — it is a **certified control layer** for conditional intervention: a gate
> with a provable false-positive bound, a displacement budget that provably limits
> collateral, and a map of when the model itself resists.

This program reframes the project from "compete on steering quality" (a crowded,
compute-heavy race a 16 GB laptop loses, and one our own data suggests is NULL at
accessible scale) to "make steering trustworthy" — which is how industry actually
thinks (Anthropic ships classifiers, not vectors, because reliability is unsolved).
It turns the project's two "embarrassing" nulls (E7 direction, N17 displacement)
into the spine of a real paper.

It unifies three independent strategic reviews forwarded into the repo
(2026-06-09/10): "conditional displacement, not direction"; "certified conditional
interventions, when/where/how-hard"; and the lab-leader 20-item hill-climb. They
agree on everything that matters, so this is one program, not three.

---

## The evidence this is built on (all SCREENING; see FINDINGS.md)

1. **E7 / S-16 / S-24 — direction is (near-)null at <=2B.** Real DiffMean beats a
   norm-matched shuffled control by +0.004 at 2B (~97% captured by the control) and
   NULL at 1B. Honest caveat (S-24): this may be *below the judge's detection floor*
   (AUC 0.68-0.74) — which is exactly why H0 (judge) gates the program.
2. **E3 — the coherence cliff** at ~5-10% relative displacement (alpha~0.05 on 1b).
3. **S-25 / S-26 — the gate is the one positive.** Detector AUC ~0.75; the raw
   cosine beats a trained probe that overfits (AUC 0.43); conditional gating saves
   off-target fluency. (The +0.125 number is WITHDRAWN pending a held-out tau.)
4. **N17 — off-manifold displacement predicts incoherence** (rho=+0.585), the
   strongest single correlation in the repo — a controller waiting to be built.
5. **S-22 — aligned models have no headroom** (gemma-2-2b ASR=0 on raw JBB), so
   safety steering must be evaluated UNDER ATTACK, or on an abliterated model
   (DavidAU/...-heretic-... complies ~100%, verified 2026-06-10) — both give headroom.

---

## The unified hypothesis registry

Stable IDs; each maps to a design doc (or is PLANNED). Status as of 2026-06-10.
All are DESIGN/PLAN — zero have been run under a calibrated judge.

| ID | Hypothesis (one line) | Design doc | Status | Phase |
|---|---|---|---|---|
| **H0** | Instrument validity: an open judge calibrates to >=0.85 AUC vs AxBench labels AND the harness reproduces one published steering result. | `A_foundations/H0_instrument_validity.md` | DESIGN; **RUNNING** (judge AUC measured this session) | 0 — foundation, blocks all |
| **H1** | Certified gate: a conformal-calibrated per-intent cosine gate gives a provable benign over-refusal bound (<=alpha) at higher recall than a same-size classifier-router. | `B_conditional/M8_conformal_gate.md` | DESIGN; RUN PENDING (blocked on H0) | 1 — the gate |
| **H2** | Direction is FUNGIBLE under the gate: with the gate fixed, real vs norm-matched random / shuffled / orthogonalized directions give equal safety-under-attack at matched coherence. (E7 null promoted to a mechanistic claim.) | `A_foundations/H2_direction_fungible_under_gate.md` | DESIGN; RUN PENDING (the decisive cheap test) | 1 — the bold null |
| **H3** | Multi-intent gates compose with bounded interference: FPR grows sub-linearly in K under Gram-Schmidt orthogonalization. | (≈ M3; PLANNED doc) | PLANNED | 1 |
| **H4** | Displacement budget controls the tax AND defends: capping off-manifold displacement (N17) holds coherence at maximal alpha and blunts the Rogue-Scalpel universal attack. | `D_geometry/H3_displacement_budget.md` | DESIGN; RUN PENDING | 2 — the budget |
| **H5** | On-manifold rescue: a cheap reprojection (Mahalanobis / kNN-bank) toward the activation distribution after the add shifts the coherence cliff rightward. | (PLANNED doc) | PLANNED | 2 |
| **H6** | Interpretable SAE-feature gates (GemmaScope) beat dense-cosine on OOD detection and emit human-readable firing reasons. | (PLANNED doc) | PLANNED | 1-2 |
| **H7** | Scale-dependent fungibility: H2's null weakens with scale (real beats controls at 9B not <=2B), tracking the model's endogenous steering-resistance rate. | `A_foundations/H4_scale_dependent_fungibility.md` | DESIGN; RUN PENDING (needs rented A100) | 3 — scale |
| **H8** | Deployment capstone: under attack, the certified gate Pareto-dominates a classifier-router on (ASR-under-attack, over-refusal, MMLU) — because it reads internal state paraphrase attacks can't evade. | (PLANNED doc) | PLANNED | 3 — capstone |

(Note: the H3/H4 design-doc *filenames* use the "displacement-plan" local numbering;
the canonical program IDs are in this table. M8 == H1.)

---

## Frozen evaluation protocol (do not change after a run starts)

| Element | Choice |
|---|---|
| Models | gemma-3-1b-it (dev), gemma-2-2b-it (main), abliterated 1b (headroom), gemma-2-9b-it (H7, rented) |
| Safety-under-attack | JBB-100 + StrongREJECT WITH the adversarial harness (PAIR/prefill/refusal-suppression) so baseline ASR is 30-60%, not 0; OR the abliterated model (~100% baseline). |
| Over-refusal | XSTest-250 (full, not n=12) |
| Capability / coherence | real MMLU, WikiText PPL |
| Judge | **calibrate to >=0.85 AUC vs AxBench labels BEFORE any run (H0)**; two judges, disagreement-flagged |
| Primary metric | raw (ASR-under-attack, over-refusal) Pareto frontier. Retire the 5-axis composite from anything public. |
| Stats | bootstrap over PROMPTS + extraction-pair resamples (greedy decoding has ~0 seed variance — NOT decode seeds); paired Wilcoxon; Holm-Bonferroni across the hypothesis family |
| Baselines | no-steer, unconditional-steer, system-prompt-refusal, CAST, **classifier-router + canned refusal** (the production baseline to beat), AlphaSteer |

---

## Sequencing (each phase gates the next)

0. **Instrument (H0)** — calibrate judge to >=0.85; reproduce one published number. *No new claim until this passes.* RUNNING this session.
1. **The gate (H1, H2, H3, H6)** — run **H2 first** (cheapest + most decisive: three controls on the existing harness tells you which paper you're writing), then the conformal gate H1.
2. **The budget (H4, H5)** — operationalize N17; red-team with the Rogue-Scalpel probe.
3. **Scale + capstone (H7, H8)** — one 9B run on a rented GPU; the deployment comparison.

## Guardrails (the reviews' "do NOT do" list)

- Do **not** chase a new vector-extraction method at small scale — the direction is null there (our own data).
- Do **not** evaluate anything with an uncalibrated judge ever again.
- Do **not** start multi-intent (H3) before single-intent gating (H1) works — breadth-before-depth was the prior failure mode.
- Do **not** publish the simulated reviews as "external"; do **not** commit generated dashboards; collapse the ledger surface (STATUS + LEDGER as the only machine-checked sources of truth).

## What success looks like

A workshop-grade paper is reachable from **H1 + H2 + H4 on 2B alone**: *"At <=2B, gated
generic displacement matches direction-specific steering for safety-under-attack, with
a conformal over-refusal guarantee and a displacement budget that doubles as a
universal-attack defense."* H7 (the 9B scale test) upgrades it to a venue paper by
adding the scaling story. If the classifier-router wins H8, the honest thesis is still
complete: *"internal gates are not yet better than text classifiers — here is the
certified framework for when they will be."*

---

## Status journal

- 2026-06-10 — Program doc created, unifying three forwarded strategic reviews. H0
  judge-calibration RUNNING; H1 (=M8), H2, H4 (displacement budget), H7 (scale) design
  docs written; H3/H5/H6/H8 registered as PLANNED. Next executable step: H0 result,
  then H2 (the decisive cheap null-promotion). The abliterated-model headroom is
  confirmed (re-alignment driver committed; a two-process split is needed to dodge a
  Windows multi-model-load crash before the quantified ASR-vs-alpha curve lands).
