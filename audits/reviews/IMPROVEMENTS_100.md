# 100 Improvements to Publishability — synthesized from 5 elite reviews

> **Goal (the north star every item serves):** a **state-of-the-art *conditional*
> activation-steering method for *multi-intent safety*** — conditionally steer an
> LLM (only when a request warrants it) toward **safer responses than the
> unsteered baseline**, across **multiple harm intents**, *without* breaking
> capability/coherence or over-refusing benign prompts. Method development on
> AxBench; **final evaluation on SOTA safety benchmarks**. Done = all five
> reviewers (ICLR / ICML / NeurIPS / elite researcher / hands-on lab leader)
> recommend ACCEPT at a top venue.
>
> **Unanimous current verdict: REJECT / 2-of-10.** The method does not exist; the
> safety axis is a regex on 10 synthetic prompts; no real safety benchmark is in
> code; the judge is AUC 0.68; zero external-ready; the one rigorous result is
> negative. Source reviews: `audits/reviews/{ICLR,ICML,NeurIPS,Researcher,LabLeader}_review.md`.

Priorities: **P0** = blocks any submission · **P1** = needed to be competitive ·
**P2** = strengthens / polish. Each item names the concrete artifact.

---

## A. THE METHOD — build the actual contribution (P0; nothing publishes without this)
1. **P0** Write `src/steering/DESIGN.md` FIRST — spec the components (Detector/intent-gate, SafetyTarget, SteeringPolicy, Guard, AdversarialHarness) + the numpy↔torch seam, before writing features.
2. **P0** Build `src/steering/cast.py` — a real in-forward conditional pipeline: read `h@L_c` → live gate decision → masked steering write `@L_b`, all in ONE forward pass (today `gate.py` is offline numpy never wired to `hooks.py`).
3. **P0** Extract and validate an Arditi-style **refusal / safe-completion direction** as a first-class `SafetyTarget` (replaces the regex notion of "safe").
4. **P0** Build the **intent gate** as a calibrated per-category linear probe on `h@L_c` over real harm categories, with per-category thresholds; report calibration (ECE) and gate PR-AUC.
5. **P0** Define + implement **multi-intent steering**: compose K safety directions and fire them independently per detected intent (the core novelty).
6. **P0** Compose the K directions via **Gram-Schmidt orthogonalization** (build E19) and measure interference, vs the naive `Σ v̂` currently used.
7. **P1** Make the gate **online / per-position**, not whole-prompt mean-pooled (real deployment reads tokens as they arrive).
8. **P1** Operationalize **N17 (off-shell ‖Δh‖)** as a coherence-budget constraint that scales/clamps the safety edit under the E3 cliff (geometry → method).
9. **P1** Use **N16 radial/angular cost** to choose the safety operation (add vs project_out vs rotate) per layer, instead of a fixed choice.
10. **P1** Fit **per-model alpha caps** from E3 (N5 failed across scale) rather than a hardcoded knee.
11. **P1** Position-aware contrast extraction (instruction vs answer-onset), not whole-prompt mean-pool.
12. **P2** Rebuild **SAE-TS faithfully** (GemmaScope + a real attribution objective) or cut E20 — the toy-SAE cannot support the claim.

## B. REAL SAFETY BENCHMARKS + A REAL SAFETY JUDGE (P0; the evaluation is currently fake)
13. **P0** Wire **JailbreakBench** end-to-end as primary attack-success-rate (ASR); fix the dashboard's mislabeled synthetic "JailbreakBench CR".
14. **P0** Wire **StrongREJECT** with its rubric grader as the headline jailbreak metric.
15. **P0** Wire **HarmBench** (+ its classifier) for breadth/cross-check.
16. **P0** Wire **AdvBench** harmful-behaviors set.
17. **P0** Wire **XSTest** as the first-class **over-refusal** axis.
18. **P0** Add **SORRY-Bench (45 categories)** / **WildGuard** / **Aegis** to operationalize multi-intent harm categories.
19. **P0** Replace the 22-string `is_refusal()` detector with a **calibrated safety classifier** (Llama-Guard-3 / ShieldGemma / HarmBench cls); require **≥0.9 human agreement on ≥100 items** (`calibration_agreement`).
20. **P0** Retire the AUC-0.68 behavior judge for any safety claim; gate every safety claim on **judge AUC ≥ 0.85** and disclose it.
21. **P0** Replace the FakeLM/MMLU surrogate with **real MMLU (≥500) + GSM8K** capability tax, measured in the same run.
22. **P0** Make coherence real (**WikiText-103 PPL**) in the same run so the composite is finally all-real (no hardcoded axes).
23. **P1** Add **OR-Bench / PHTest** for population-scale over-refusal (beyond 8 prompts).
24. **P1** Pin a **dataset/version manifest** (source, sha256, split, seed, license) for every benchmark + retire the `*_mini.json` slices as claim instruments.
25. **P2** Add **MT-Bench / AlpacaEval** helpfulness panel — the deployability case for *conditional* (not always-on) steering.

## C. DEFINE & OPERATIONALIZE "MULTI-INTENT" (P0; currently undefined anywhere)
26. **P0** Formally **define multi-intent** + pre-register it: K≥2 simultaneous harm categories / compositional (benign-wrapped) jailbreaks.
27. **P0** Build a **multi-intent eval set** (co-occurring benign + harmful intents; dual-intent prompts).
28. **P0** Report **per-category ASR + joint over-refusal** as the multi-intent success metric (not a mean).
29. **P1** OR-gating coverage (E11): coverage scales across ≥5 condition vectors while XSTest over-refusal stays flat.
30. **P1** Real multi-category condition-vector **orthogonality** (E10) on SORRY-Bench categories (not the toy emotion concepts).
31. **P2** Compositional jailbreaks (benign wrapper hiding harmful intent) to show activation-conditioning beats token-filtering.

## D. BASELINES — the comparisons that decide acceptance (P0/P1)
32. **P0** No-steer baseline (the true alpha=0 reference; current safety baseline is an instrument artifact).
33. **P0** **Prompting / few-shot safety** baseline (AxBench shows prompting beats steering — must be confronted head-on).
34. **P0** **System-prompt refusal** baseline.
35. **P0** **Unconditional steering** baseline (so "conditional" is isolated — gate-on vs gate-off is THE central ablation).
36. **P0** Published **CAST** baseline (the method being extended).
37. **P1** **Refusal-direction (Arditi)** ablation/activation baseline.
38. **P1** **Llama-Guard / ShieldGuard router** baseline (a second-model classifier gate) + the latency/compute comparison vs activation gating (the claimed selling point).
39. **P1** **Circuit-breakers / RepE safety** baseline.
40. **P1** **Safety SFT / DPO** baseline (the "just fine-tune it" rebuttal).
41. **P1** Report the **core Pareto frontier**: ASR (JailbreakBench/StrongREJECT) vs over-refusal (XSTest/OR-Bench), with all prior methods plotted.
42. **P1** Single **headline metric**: ASR reduction at fixed ≤1% over-refusal and ≤2pp MMLU drop, vs each baseline.

## E. STATISTICAL RIGOR — fix the contract (P0/P1; several legs are currently broken)
43. **P0** Fix the **Holm leg**: `rigor_report(family_pvalues=None)` makes Holm pass vacuously on every AxBench driver — apply Holm across the real family.
44. **P0** Control multiple comparisons at the **program level** (70-hypothesis garden of forking paths) — a held-out confirmation set + a pre-committed promotion budget.
45. **P0** Fix the **composite-integrity bug**: AxBench drivers log `mean_delta` into the `composite` field under fingerprint `a9001e87087e` (two different numbers, one fingerprint).
46. **P0** **Pre-register** the minimum meaningful effect (MME) per axis; gate verdicts on `effect ≥ MME AND p<α AND correct sign` — never p alone.
47. **P0** Enforce in code: n=3 → SCREENING (no p-claims); **n≥7 → EVALUATION**; the runner refuses "SUPPORTED/DIRECTIONAL" at n<7.
48. **P1** Re-specify the **ordinal gate** for the concept-as-replicate design (per-concept paired/sign test, not worst-vs-best across heterogeneous concepts).
49. **P1** Run **≥7-seed variance** for the real pipeline (extraction/judge/batching) and enforce the empirical 2σ_seed band.
50. **P1** Bound the **coherence tax** (`dppl_norm` is unbounded → composite becomes a coherence-explosion detector) or report the 5-axis Pareto directly.
51. **P1** **λ-weight sensitivity**: re-run verdicts at λ ±50%; show rankings survive reweighting.
52. **P1** Promote **effect sizes** (% of control captured, Cliff's δ, CIs) to the primary reported number alongside p.
53. **P2** Pre-register the **full safety eval plan** (benchmarks/judge/seeds/thresholds/baselines) in git BEFORE running (anti-HARKing, given how E7 played out).

## F. ADVERSARIAL ROBUSTNESS / RED-TEAM (P1; safety claims require attack)
54. **P1** Adaptive attacks: **GCG** suffixes.
55. **P1** **PAIR** / **AutoDAN** automated jailbreaks.
56. **P1** **Prefilling** attacks.
57. **P1** Reproduce the **Rogue-Scalpel 20-vector universal attack** as a red-team probe; the guarded method must neutralize it (ACCEPT gate).
58. **P1** Report **ASR-under-attack** (not just clean ASR) as a headline.
59. **P1** Implement + ablate the **five-layer guard A–E** (subspace lock / norm clamp / mid-layer avoidance / dual-forward verdict / conditional gate).
60. **P2** GCG **suffix transfer** check across models.

## G. SCALE & EXTERNAL VALIDITY (P1; E7 showed safety effects are scale-dependent)
61. **P1** Run the headline safety result on **Gemma-2-2B-it** end-to-end (the canonical model).
62. **P1** Scale ≥1 safety result to **Gemma-2-9B-it** (4-bit fits 16 GB) — toy-model results will be dismissed.
63. **P2** **IT→base transfer** (E8) of the safety method.
64. **P2** **Cross-family** demonstration (≥2 model families) before any SOTA claim.
65. **P2** 4-bit ↔ fp16 invariance (E5) before claiming any transfer result.

## H. TRANSPARENCY & OUTCOME ARTICULATION (P0; the author's central complaint)
66. **P0** Rewrite the **README first screen** to the real goal + an honest status ("method unbuilt; zero external-ready") — readable in 30 seconds.
67. **P0** Add a README **"Outcome & success criterion" box**: `<metric>` on `<benchmark>` beating `<baseline>` by `<margin>` = success; show **current vs target**.
68. **P0** Fix the **README ↔ audit contradiction**: README says "unconditional ICML sign-off" while `ICML_SIGNOFF_v2.md` says "conditional accept, blocker open."
69. **P0** Remove **self-grading** banners ("Rubric E 8/8", "unconditional sign-off") from README + dashboard.
70. **P1** Add **`STATUS.md`** as the single source of truth: per-claim `built? / tested? / validated? / result / instrument`.
71. **P1** Add a **method-ladder doc**: refusal-dir → in-forward gate → multi-intent → over-refusal control → adversarial → SOTA, each with a promotion gate.
72. **P1** Every dashboard/result cell carries **instrument provenance**: `safety_real`, judge id + AUC/κ, n, seeds, model, layer.
73. **P2** Pre-register the safety success criterion in git before the CAST sweep.

## I. ENGINEERING / REPRODUCIBILITY / DESIGN DOCS (P0/P1)
74. **P0** `scripts/reproduce_headline.sh` — ONE command runs the method + baselines + benchmarks + rigor + dashboard, prints PASS/FAIL vs the pre-registered threshold.
75. **P0** Add **CI** (ruff + mypy + pytest + verify_rubrics + fingerprint) — would have caught the dashboard lint/type regressions.
76. **P0** Stop presenting `best_config.json` **safety fields as measured** when they're hardcoded defaults (null/N-A them).
77. **P1** Split the **4,153-line `dashboard.py`** into modules; clear flagged lint/type errors; freeze feature growth.
78. **P1** Add **`ARCHITECTURE.md`** with a request → gate → steer → guard → judge diagram (PNG).
79. **P1** **Dataset manifests** (source/sha/seed/license) for all data; bump the judge-cache namespace whenever the judge changes.
80. **P2** Add a `make repro` / task runner for canonical commands.
81. **P2** **Quarantine FakeLM-only "results"** from all results-facing surfaces.

## J. ANTI-SLOP — cut volume-over-substance (P0/P1; the "AI slop" complaint)
82. **P0** Demote the ~50 **UNTESTED** hypothesis docs to `backlog/`; stop counting them as deliverables.
83. **P1** Prune **decision-free dashboard cells** (α≥4 garbage rows, composite −2.5M, PPL 5e6) — every cell must inform a decision.
84. **P1** Collapse the dashboard to a **decision-first front page**; stop generating result-shaped pages for hypotheses with no data.
85. **P1** Cut **FINDINGS.md ~60%**: reach the ~7 real observations in under a page.
86. **P1** **Leakage check**: harmful/benign eval prompts must not share surface tokens (the gate-as-keyword-classifier confound).
87. **P2** Consolidate the **five audit/sign-off files** into one; add the same-model-family circularity disclaimer.
88. **P2** Audit every **arXiv ID**; flag/remove unverifiable citations.

## K. PER-HYPOTHESIS DEPTH / MECHANISM (P1/P2; the "shallow" complaint)
89. **P1** Add real **mechanism** work to the safety hypotheses: activation patching, logit-lens, per-layer **Fisher** on real safety contrasts (not asserted prose).
90. **P1** Re-scope the hypothesis registry around the deliverable; demote the now-null E2/E4/E27/E36 to an appendix.
91. **P2** Each kept hypothesis page: real per-experiment provenance + the actual mechanism finding (not templated filler).
92. **P2** Connect the **geometry strengths (N17/E3/N16)** explicitly into the safety method (stress-test the norm-budget guard on safety stacks).

## L. PAPER & FRAMING (P1)
93. **P1** Rewrite the paper **around the method**; new title; delete the "Reviewer-loop changelog" appendix.
94. **P1** State the **contribution** in one sentence in the abstract (currently the paper says "does not propose a new steering method").
95. **P1** Position **novelty** empirically vs CAST/RepE: composability / latency / multi-intent / black-box — pick the axis and prove it head-to-head.
96. **P1** Write **Broader Impact / dual-use + responsible disclosure** (the Rogue-Scalpel inversion: a safety steer is an unsafe steer flipped).
97. **P2** Report the helpfulness/deployability case for *conditional* over always-on steering.
98. **P2** Confront the AxBench negative explicitly in related-work/limitations (don't ship "SOTA steering" on null data).

## M. FALLBACK & SEQUENCING (P1)
99. **P1** Pre-commit a **fallback paper**: if the safety method doesn't beat baselines, ship the honest **mechanisms paper** (N17 off-shell→incoherence + E3 cliff + the E7 "direction is ~97% generic" negative) — the repo's strongest current asset, a real workshop/findings contribution.
100. **P0** **Sequencing/exit-criteria**: do A+B+H+the DESIGN.md first (2-week P0 sprint), then run the method-vs-baselines Pareto on real benchmarks; only expand (F/G/K) once the headline clears its pre-registered threshold. A rigorous negative is acceptable; an unmeasured claim is not.

---

## Definition of done, per reviewer
- **ICLR:** a named, implemented method; real safety-benchmark eval; Pareto-dominance over CAST + prompting at n≥7; clear contribution; no slop/self-grading.
- **ICML:** ≥0.9-agreement safety classifier; powered n with correct Holm/ordinal/MME; full baseline suite; ablations (gate on/off); honest effect sizes.
- **NeurIPS:** significant + novel safety contribution; multi-intent defined + measured on SOTA benchmarks; adversarial robustness; broader-impact/dual-use.
- **Researcher:** a genuinely novel conditional/multi-intent mechanism (not DiffMean re-run); geometry findings feeding the method; DESIGN.md; depth per hypothesis.
- **Lab leader:** 30-second-clear goal/outcome/status; one-command reproduce; CI; design docs; zero slop; results that are real, not hardcoded.
