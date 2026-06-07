# STATUS — Single Source of Truth

> Read this first. A reviewer should be able to assess project state in 2 minutes.
>
> Updated: 2026-06-06. For the full rigor contract see FINDINGS.md. For the
> improvement roadmap see audits/reviews/IMPROVEMENTS_100.md.

---

## Headline

**Method newly built, not yet validated. Zero external-ready results.**
Five external reviewers returned a unanimous reject (2/10). The methodology and
reproducible harness are sound (Rubric E 8/8 after fixing a lint/type blocker).
The safety method and its benchmarked results are the gap.

---

## Per-claim / per-component table

| Component | Built? | Unit-tested? | Validated on real benchmark? | Result / Instrument |
|---|---|---|---|---|
| Shared harness (hooks, extract, geometry, eval, runner) | YES | YES (46 tests, Rung 0) | N/A (infrastructure) | pytest green |
| Composite formula (fingerprinted) | YES | YES | N/A | fingerprint a9001e87087e |
| Dashboard generator (dashboard.py) | YES | YES (Rubric B 15/15) | N/A | ruff + mypy clean |
| Refusal direction (SafetyTarget) | PARTIAL (cast.py exists) | NO | NO | method component not yet wired end-to-end |
| Intent gate (per-category calibrated probe) | NO | NO | NO | missing; gate.py is offline numpy not wired to hooks.py |
| Multi-intent steering (compose K safety directions) | NO | NO | NO | missing |
| JailbreakBench / StrongREJECT wiring | NO | NO | NO | safety metric is synthetic 10-prompt regex; labeled mislabeled "JailbreakBench CR" in dashboard |
| XSTest (over-refusal axis) | NO | NO | NO | missing |
| HarmBench / AdvBench | NO | NO | NO | missing |
| Calibrated safety classifier (Llama-Guard-3 / ShieldGemma) | NO | NO | NO | current is_refusal() is 22-string regex |
| Behavior judge (off-family LLM) | PARTIAL | PARTIAL | PARTIAL | Qwen2.5-7B-Instruct local, AUC 0.68 vs AxBench ground truth — below 0.80 bar |
| Real MMLU capability tax (in same run) | NO | NO | NO | composite MMLU axis is FakeLM surrogate |
| Real coherence (WikiText-103 PPL, in same run) | NO | NO | NO | current is wikitext_ppl_mini |
| E3 alpha-coherence cliff | YES | YES | YES (AxBench, 30 concepts) | Supported: behavior + coherence peak alpha~0.10, super-linear collapse past ~0.20; judge AUC 0.68 |
| N17 off-shell displacement predicts incoherence | YES | YES | YES (WikiText-2, n=50, non-iid) | Supported: Spearman +0.585, CI [0.35, 0.76], p=8e-6; non-iid caveat |
| E7 directional steering (DiffMean vs shuffled) | YES | YES | YES (AxBench, n=500 concepts) | WEAK/negative: +0.004 at 2B (ordinal gate fails; ~97% captured by shuffled); NEGATIVE at 270M |
| E2 layer sweep | YES | YES | YES (AxBench, 20 concepts, 2B) | Nearly flat: behavior range 0.163-0.184 across layers 6-22 |
| E4 DiffMean vs PCA-top1 alignment | YES | YES | YES (AxBench, 100 concepts, 2B) | Moderate on real concepts: mean |cos| 0.65; synthetic cos~0.99 did not generalize |
| E27 rotation vs addition | YES | YES | YES (AxBench, 20 concepts, 2B) | No benefit: rotate-vs-add delta -0.003; not supported |
| E36 PCA vs DiffMean source | YES | YES | YES (AxBench, 20 concepts, 2B) | Wash: +0.009 behavior PCA but -0.037 coherence; source barely matters |
| E17 two-vector stacking | YES | YES | NO (synthetic only) | Supported screening: retains 101%/110% on synthetic anger+happiness |
| N5 universal norm-budget law | YES | YES | YES (held-out WikiText-2) | FALSIFIED across scale: held-out R²=-1.6 when predicting 1B from 270M coefficients |
| E15 learned gate vs fixed threshold | YES | YES | NO (tiny OOD dataset) | FALSIFIED_OOD: logistic gate overfits, OOD AUC 0.55 vs cosine threshold 0.72 |
| E45 HyperSteer | YES | YES | NO | INCONCLUSIVE: LOO cos~0, proxy unreliable |
| Adversarial evaluation (GCG / PAIR / AutoDAN) | NO | NO | NO | missing |
| Rogue-Scalpel 20-vector universal attack red-team | NO | NO | NO | missing |
| Five-layer guard (A-E: subspace lock / norm clamp / mid-layer avoidance / dual-forward / conditional gate) | NO | NO | NO | not implemented |
| Prompting / few-shot safety baseline | NO | NO | NO | missing; AxBench shows prompting competes with steering |
| CAST baseline | NO | NO | NO | missing |
| Unconditional steering baseline | NO | NO | NO | missing |
| Gram-Schmidt orthogonalization (E19) | NO | NO | NO | missing |
| n>=7 seed evaluation (any axis) | NO | NO | NO | all evaluations are n=1 to n=20 synthetic or n=500 concepts under weak judge |
| Statistical rigor (paired Wilcoxon + bootstrap CI + Holm-Bonferroni + ordinal gate) | PARTIAL | PARTIAL | PARTIAL | contract implemented in stats.py; Holm leg passes vacuously when family_pvalues=None |

---

## External-ready findings

**Zero.** No result meets the full six-part rigor contract (n>=7 iid seeds, paired
Wilcoxon p<0.05, bootstrap CI excluding zero, Holm-Bonferroni correction, ordinal
gate, logged at rung 3+ in EXPERIMENT_LEDGER.md). All S-1..S-21 observations are
SCREENING ONLY and cannot be cited as research findings.

---

## What the safety method needs before it can be evaluated

The minimal P0 sprint (see audits/reviews/IMPROVEMENTS_100.md items A-H):

1. Wire cast.py as a real in-forward conditional pipeline (read h@L_c -> gate -> steer @L_b in ONE forward pass).
2. Extract and validate an Arditi-style refusal direction as the SafetyTarget.
3. Wire JailbreakBench end-to-end (replace the synthetic regex safety axis).
4. Wire XSTest as the over-refusal axis.
5. Replace the AUC-0.68 behavior judge with a calibrated judge (AUC >= 0.85).
6. Run the method vs CAST baseline and a prompting baseline on the Pareto frontier.
7. Pre-register the success criterion in git before running.

---

## Reviewer status

| Reviewer | Verdict | Score | Key complaint |
|---|---|---|---|
| ICLR | Reject | 2/10 | Method does not exist; safety axis is a regex |
| ICML | Conditional accept (blocker: lint/type) | -- | Blocker fixed; methodology sound |
| NeurIPS | Reject | 2/10 | Judge AUC 0.68; zero external-ready; prompting baseline missing |
| Elite researcher | Reject | 2/10 | The one rigorous result is negative; no SOTA claim possible |
| Lab leader | Reject | 2/10 | No multi-intent definition; no real safety benchmark wired |

Roadmap: audits/reviews/IMPROVEMENTS_100.md (100 items; P0 sprint = groups A+B+H).
