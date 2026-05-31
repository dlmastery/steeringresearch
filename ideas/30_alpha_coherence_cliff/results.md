# E3 — Alpha coherence cliff: results (SCREENING, n=1)

## ⭐ REAL-INSTRUMENT RE-RUN (supersedes the projection-proxy table below)

After the ICML review (`audits/ICML_REVIEW.md`) flagged the projection-proxy as
circular and safety as stubbed, both instruments were made real: behavior =
**generation-based concept incorporation** (concept words in *generated text*,
steered vs unsteered — never the injected vector); safety = **real steered
generations** scored by the refusal detector. Model: Qwen2.5-0.5B-Instruct
(bring-up surrogate; **Gemma reproduction pending model download** — NEXT_STEPS).
exp#10–14, tags `E3real-cliff-a*`. n=1 SCREENING.

| α | behavior (generation) | PPL | compliance_rate | composite |
|---|---|---|---|---|
| 0 | 0.500 | 48.9 | 0.30 | −0.107 |
| 1 | **0.694** ← peak | 58.7 | 0.30 | **−0.073** ← best |
| 2 | 0.526 | 89.0 | 0.30 | −0.717 |
| 4 | 0.494 | 293.6 | **0.60** | −3.597 |
| 8 | 0.346 | 3 787 | **1.00** | −40.602 |

**Two findings the circular proxy completely hid:**
1. **Behavior is non-monotonic and peaks at α≈1, then DECLINES** (0.69→0.53→
   0.49→0.35). Over-steering destroys not just coherence but the *behavior
   itself* — incoherent text cannot express the concept. The proxy reported a
   flat saturated 1.0 and missed this entirely. The behavior/coherence sweet spot
   is α≈1 (also the composite optimum).
2. **Steering compromises safety — the Rogue-Scalpel effect, now measured.**
   Real compliance_rate rises **0.30 → 0.60 → 1.00** as α goes 0→4→8: steering
   degrades refusal, and at α=8 the model complies with 100% of the (synthetic)
   harmful prompts. `safety_real=True`; the `λ_safe` gate now fires.

*Caveat: baseline CR=0.30 reflects weak synthetic jailbreak_mini prompts + a
non-safety-tuned 0.5B model; the DIRECTION (steering↑ ⇒ CR↑) is the real signal,
the magnitude is not transferable. Real JailbreakBench + a calibrated judge +
Gemma + n≥7 remain the gate to any external claim.*

---

## (Superseded) projection-proxy run — kept for provenance

**Status:** SCREENING observation (n=1, non-Gemma). **NOT an external claim.**
**Model:** Qwen2.5-0.5B-Instruct (non-gated 0.5B; corpus's non-surjectivity
reference model). **Layer:** 21 (max-Fisher). **Behavior vector:** DiffMean over
the 10 axbench_mini contrast pairs. **Composite fingerprint:** a9001e87087e.
**Experiments:** #2–#9 in `experiment_log.jsonl` (tags `E3-cliff-a*`).

## The curve

| α | behavior* | PPL | ΔPPL_norm | MMLU-tiny drop | off-shell Δ‖h‖ | composite |
|---|---|---|---|---|---|---|
| 0 | 0.50 | 48.9 | 0.00 | 0.00 pp | 0.000 | +0.493 |
| 1 | 1.00 | 58.7 | +0.20 | +0.05 pp | 0.033 | **+0.834** |
| 2 | 1.00 | 89.0 | +0.82 | +0.20 pp | 0.102 | +0.357 |
| 4 | 1.00 | 293.6 | +5.01 | +0.30 pp | 0.318 | −1.891 |
| 8 | 1.00 | 3 787 | +76.5 | +0.45 pp | 0.922 | −37.9 |
| 12 | 1.00 | 39 617 | +810 | +0.45 pp | 1.625 | −404.8 |
| 16 | 1.00 | 251 483 | +5 146 | +0.50 pp | 2.359 | −2 573 |
| 24 | 1.00 | 5.18 M | +106 081 | +0.50 pp | 3.875 | −53 041 |

\* behavior = the **projection proxy** — see the critical caveat below.

## What the data supports (mechanism, not magnitude)

1. **The coherence cliff is real and super-linear (E3 supported).** PPL grows
   faster than linearly in α: +20% → +82% → 6× → 77× across α=1→2→4→8. Clear knee
   around **α≈1–2**; below it PPL is near-flat, above it PPL explodes. The
   composite, pricing coherence, peaks at **α≈1** then collapses — the
   fingerprinted multi-objective metric behaves as designed.
2. **Off-shell displacement tracks incoherence (N17 supported).** Δ‖h‖ rises
   monotonically (0.033→0.102→0.318→0.92→…→3.88) in lockstep with PPL — the cheap,
   behavior-free geometry probe is a valid leading indicator of the cliff.
3. **On this tiny MMLU proxy capability is NOT the binding constraint** — MMLU
   drop stays <0.5 pp even at α=24. This is a **limitation of `mmlu_tiny`**, not
   evidence of capability robustness.

## Critical caveats (from `audits/ICML_REVIEW.md`)

- **Behavior is a circular proxy** ≈ logistic(α·‖v‖): it saturates to 1.0 by α=1
  and measures *that addition adds*, not that behavior changed. **No efficacy
  claim is made.** Real generation-based concept incorporation is the immediate
  next step (crux fix in progress).
- **Synthetic data.** The *shape* of the cliff is trustworthy; the *absolute*
  PPL/α values are not transferable to Gemma or real benchmarks.
- **n=1, non-Gemma ⇒ SCREENING.** External claims require n≥7 + the rigor
  contract + real AxBench/MMLU/WikiText + a prompting baseline + Gemma
  reproduction.

## Verdict

E3's **mechanism** (a super-linear coherence cliff with an identifiable knee,
predicted by off-shell displacement) is **supported as a screening observation**,
consistent with manifold-curvature first principles and N17. Promotion to a
DEV/STANDARD claim is gated on the instrument fixes above.
