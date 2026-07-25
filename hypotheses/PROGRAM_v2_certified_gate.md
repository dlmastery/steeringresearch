# Research Program v2 — The Certified Gate

> **Supersedes** `PROGRAM_conditional_displacement.md` (2026-06-10). That program was
> written before the June–July 2026 literature landed. Three of its assumptions are
> now falsified by publication, not by experiment. This revision keeps the one leg
> that is still open, demotes one that was scooped, and reframes one that was scooped
> unconditionally but survives in a conditional form.
>
> Written 2026-07-25 from `corpus/LIT_2026-07_*.md` (all ids fetch-verified unless
> marked `[UNVERIFIED]`).

---

## 0. What changed, and why the old program cannot be run as written

| v1 leg | v1 role | 2026-07 status | Why |
|---|---|---|---|
| Conformal gate + over-refusal bound (H1) | supporting | **OPEN — now the headline** | Nothing applies distribution-free conformal calibration to an *activation-space* safety gate with an over-refusal bound. |
| Displacement / norm budget (H4) | Phase-2 pillar | **SCOOPED — demote to a control variable** | `2602.02712` (v2, 8 Jul 2026) owns strength→cross-entropy degradation across 11 models; `2606.06735` owns the norm-vs-angle decomposition and the norm→stability tradeoff; `2604.09839` *proves* the off-manifold premise we intended to assume. |
| Direction fungibility (H2) | "the bold null", run first | **SCOOPED unconditionally — reframe** | Published three times: Rogue Scalpel `2509.22067` (random directions compose into a universal attack), Non-Identifiability `2602.06801` (behaviourally indistinguishable equivalence classes), and `2606.20852` which ran *our exact protocol* (20 norm-matched random vectors, layer-matched, bootstrap CI). |

**Consequence for N17.** Our strongest surviving in-house result (off-shell
displacement ↔ incoherence, ρ=+0.585) is no longer a novelty claim. It remains a
*correct, independently-reproduced* measurement and a useful internal instrument —
but it must be cited as corroboration of `2602.02712` / `2606.06735`, never as a
discovery.

**Consequence for the READ≫WRITE asymmetry** (diff-of-means fails to restore refusal
on abliterated Gemma-3-1B *and* 4B while a probe on the same activations separates at
AUC ≈0.99): this is the **predicted** outcome as of 2 Jul 2026 under the
multi-dimensional-refusal-subspace view (`2607.02714`). Position it as *confirmation
at a model tier that paper does not cover, plus quantification of an asymmetry
`2606.29441` implies but never isolates.* **Never present it as a surprise.**

---

## 1. The revised thesis

> **A calibrated gate — not the written direction — carries the information in
> conditional activation steering; and the gate's false-positive behaviour can be
> given a distribution-free guarantee.**

The contribution is the *composition*: a conformal-calibrated activation-space gate,
an over-refusal bound that holds under stated exchangeability assumptions, and an
empirical demonstration under attack. Every ingredient exists; the assembly does not.

## 2. Sequence (each phase gates the next)

### Phase 0 — H0: the instrument. **Blocks everything.**
No claim may be made on an uncalibrated judge. This is the failure that invalidated
the previous 124 experiments (judge AUC 0.68 against a self-imposed ≥0.85 bar).

- Calibrate **Qwen3-4B-Instruct-2507** against AxBench ground-truth labels using the
  existing `scripts/validate_judge.py` (`--judge-model`).
- **Gate: ROC-AUC ≥ 0.85.** On failure, escalate to Qwen2.5-7B-Instruct and re-run.
- Report the judge's own CI; record the judge id + AUC in every downstream row.
- Deliverable: `hypotheses/A_foundations/H0_instrument_validity.md` (referenced by v1
  as blocking, but never written) + a logged calibration result.

### Phase 1 — H1: the certified gate. **The headline.**

**Pre-flight: RESOLVED 2026-07-25 — the threat is refuted, and the framing improves.**
`2603.14623` is *"Proactive Routing to Interpretable Surrogates with Distribution-Free
Safety Guarantees"* (Uddin, Khider & Bauer, 15 Mar 2026; fetch-verified). It does **not**
claim guarantees hold for any score regardless of quality. It derives a **feasibility
condition** linking safe routing to the base safe rate π and the risk budget α, together
with **sufficient AUC thresholds under which feasible routing exists**; calibration
"primarily affects routing efficiency rather than distribution-free validity."

Two consequences, both favourable:
1. **Our theory contribution does not collapse.** The paper is about routing to
   interpretable surrogates, not an activation-space steering gate, and its guarantee is
   explicitly *conditional on gate discriminative power*. Cite it as the nearest
   formalism and state the delta (activation-space gate; controlled quantity is
   over-refusal on a safety intervention).
2. **Our READ≫WRITE asymmetry becomes load-bearing rather than merely negative.** A
   probe reaching AUC ≈0.99 on the same activations where the WRITE fails is exactly the
   regime in which this feasibility condition is *satisfied*. The measured gate quality
   is the precondition that makes a certified gate possible at all — so H1 should
   **report the gate's AUC against the paper's sufficient-AUC threshold** as a first-class
   result, not as background.
- Split-conformal calibration of the activation cosine gate on the abliterated target;
  target over-refusal α; measure realised harmless-refusal rate vs the bound.
- Baseline to beat: `2605.28722`'s energy-calibrated applicability gate (uncalibrated,
  no guarantee) and a same-size text classifier-router.
- **Pre-register as limitations, not discoveries:** (a) exchangeability is broken by
  obfuscated-activation attacks (`2412.09565`), so the bound is conditional on a
  non-adaptive adversary; (b) calibration is void after a model update (`2606.15980`).
- Cite `2606.12299` (conformalized intervention gate for VLAs) *first* and state the
  delta — it is the paper a reviewer will raise.

### Phase 2 — H2′: fungibility **under the gate** (reframed).
The unconditional question is settled. The open question is:

> With a calibrated gate fixed, is the written direction interchangeable with a
> norm-matched random one **at matched coherence**?

- Use the `2606.20852` protocol exactly: norm-matched, layer-matched, ≥20 random
  controls, bootstrap CIs.
- Report **per-model and per-strength** — `2602.06801` and `2605.27681` `[UNVERIFIED]`
  both show fungibility is model- and strength-dependent, not universal.
- The abliterated refusal-restoration setting (where the direction demonstrably fails)
  is unoccupied and is our natural venue.
- Framing: *"under a calibrated gate the direction is a knob and the gate carries the
  information"* — **never** *"steering directions are arbitrary"* (published ×3).

### Phase 3 — **matched-budget attribution** (the budget leg, rescued and narrowed).

The geometry scan (`corpus/LIT_2026-07_geometry.md`) confirms the budget-as-controller
idea is closed: `2601.19375` (*Selective Steering: Norm-Preserving Control Through
Discriminative Layer Selection*, Dang & Ngo, 27 Jan 2026 — eight models including **four
in our 1B–4B band**, code public) ships norm-preserving control with zero PPL-threshold
violations; `2606.19946` (GEMS), `2605.05115` (on-manifold projection) and `2510.04309`
(PID Steering, closed-loop) close the remaining variants.

**Stop calling the displacement budget novel.** Three gaps survive, and two are ours:

1. **No matched-budget comparison exists.** Selective Steering compares three *rotation*
   methods to each other and dismisses additive steering as "catastrophic degradation on
   smaller models" **without a controlled equivalence test** — it never holds displacement
   fixed and asks whether the rotation path is better *per unit of displacement spent*.
   **We already ran that experiment.** At matched norm budget the curved/rotated path was
   *worse*, not better (harmful gibberish 0.77 vs 0.49 for the straight chord; measured
   arc length = chord length, off-shell 1.1e-16 vs 0.076). This directly contradicts the
   field's working assumption in the exact hole it left. **This is now a real leg.**
2. **Nobody uses displacement as a calibrated *online* predictor.** Published methods
   constrain geometry a priori (rotate / project / clamp). None measures per-token
   displacement at inference and modulates against a *calibrated* coherence setpoint —
   PID Steering is closed-loop but its feedback signal is the steering vector itself, not
   a coherence proxy. This is where N17 survives: not as an observation, but as the
   calibrated signal inside a controller.
3. The angle/norm decomposition (`2606.06735`) is asserted, not measured as a controller,
   and not reported at ≤4B.

**Reframe:** the contribution is *matched-budget attribution* — which component of
displacement (radial vs angular) carries the coherence tax at ≤4B — citing `2601.19375`,
`2606.19946`, `2605.05115`, `2510.04309` as the methods we decompose, **not** as ideas we
invented. Pre-register the matched-budget equivalence test as the primary experiment.

---

## 3. Frozen protocol (unchanged from v1 except where noted)

| Element | Choice |
|---|---|
| Target | abliterated Gemma-3-1B (dev) → Gemma-3-4B (confirm). Aligned models have ASR=0 and no headroom (S-22). |
| Judge | **Qwen3-4B-Instruct-2507, calibrated ≥0.85 AUC before any run** (H0). |
| Controls | ≥20 norm-matched, layer-matched random directions + label-shuffled, per `2606.20852`. |
| Statistics | bootstrap over **prompts and extraction pairs** (greedy decoding has ~0 seed variance); paired Wilcoxon; Holm across the hypothesis family. |
| Primary metric | raw (ASR-under-attack, over-refusal) Pareto frontier. **The 5-axis composite is retired from anything public** — see `autoresearch_results/PROVENANCE.md`. |
| Baselines | no-steer, unconditional-steer, CAST, classifier-router + canned refusal, energy-gate (`2605.28722`). |

## 4. Guardrails (inherited + new)

1. Never evaluate with an uncalibrated judge. *(This is why v1's evidence base failed.)*
2. Never run on a substrate without headroom, or below 1B.
3. A screening run is not a result: n and the statistical contract are declared before the run.
4. **New:** before any leg is promoted to "contribution", re-run the novelty check —
   two of three v1 legs were scooped within six weeks. Novelty decays; verify late, not once.
5. **New:** cite the nearest competing work *first*, and state the delta explicitly.

## 5. Status journal

- **2026-07-25** — v2 written. Judge downloaded. H0 unblocked and is the next executable
  step. Pre-flight verification of `2603.14623` must precede any Phase-1 compute.
