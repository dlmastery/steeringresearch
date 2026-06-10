# STATUS — Single Source of Truth

> Read this first. A reviewer should be able to assess project state in 2 minutes.
>
> Updated: 2026-06-06. For the full rigor contract see FINDINGS.md. For the
> improvement roadmap see audits/reviews/IMPROVEMENTS_100.md.

---

## Headline

**Method + full evaluation harness now BUILT and unit-tested; NOT yet run on real
models/benchmarks. Zero external-ready results.**
Five SIMULATED adversarial reviews (LLM role-played in-session — NOT human peer
review or any venue) returned a unanimous reject (2/10). Waves A/B/C of the
100-item roadmap are built (commits bd5d4e7, 5223300): the conditional
multi-intent method (cast.py), the Qwen-7B safety judge, the real safety-benchmark
loaders, the 7 baselines, the adversarial red-team harness, the end-to-end driver,
and the rigor-bug fixes — all offline-tested (~260 tests green, ruff + mypy clean).
**The remaining gap is RUNS, not code:** free VRAM, calibrate the Qwen-7B judge,
verify the benchmark HF ids, then climb the method ladder (docs/METHOD_LADDER.md).

---

## Per-claim / per-component table

| Component | Built? | Unit-tested? | Validated on real benchmark? | Result / Instrument |
|---|---|---|---|---|
| Shared harness (hooks, extract, geometry, eval, runner) | YES | YES (46 tests, Rung 0) | N/A (infrastructure) | pytest green |
| Composite formula (fingerprinted) | YES | YES | N/A | fingerprint a9001e87087e |
| Dashboard generator (dashboard.py) | YES | YES (Rubric B 15/15) | N/A | ruff + mypy clean |
| Refusal direction (SafetyTarget) | YES (safety_target.py) | YES | NO | extract_refusal_direction (DiffMean); real extraction on Gemma not yet run |
| Conditional method (cast.py: in-forward gate->masked steer) | YES | YES (conditional-identity test) | NO | CASTSteerer + UnconditionalSteerer; real Gemma run PENDING |
| Intent gate (per-category calibrated probe) | YES (intent_gate.py) | YES | NO | target-FPR thresholds + ECE; real calibration data not yet run |
| Multi-intent steering (compose K safety directions) | YES (multi_intent.py) | YES | NO | Gram-Schmidt + interference Gram mass |
| JailbreakBench / StrongREJECT wiring | YES (safety_bench.py) | YES (mocked) | NO | loaders built; HF ids flagged [VERIFY]; real download not yet run |
| XSTest (over-refusal axis) | YES (safety_bench.py) | YES (mocked) | NO | loader built; over_refusal_rate in safety_judge |
| HarmBench / AdvBench / SORRY-Bench | YES (safety_bench.py) | YES (mocked) | NO | loaders built; HF ids flagged [VERIFY] |
| Safety classifier / judge (Qwen-7B, NOT Llama-Guard) | YES (safety_judge.py) | YES (mocked) | NO | Qwen2.5-7B safety grader (ASR/over-refusal/calibrate); calibration run PENDING |
| Behavior judge (off-family LLM) | YES (local_judge.py) | PARTIAL | PARTIAL | Qwen2.5-7B local, AUC 0.68 vs AxBench ground truth — below 0.80 bar; re-calibration PENDING |
| Real MMLU/GSM8K/WikiText metrics | YES (real_metrics.py) | YES | NO | replaces FakeLM surrogate; real run PENDING |
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
| Adversarial harness (Prefill/RefusalSuppression/Roleplay/Encoding/PAIR/GCG templates) | YES (adversarial.py) | YES (18 tests) | NO | prompt-space transforms; real GCG/PAIR optimization PENDING (needs model in loop) |
| Rogue-Scalpel 20-vector universal attack red-team | YES (RogueScalpelProbe) | YES | NO | probe + evaluate_under_attack built; live run PENDING |
| Five-layer guard (A-E: subspace lock / norm clamp / mid-layer avoidance / dual-forward / conditional gate) | PARTIAL | PARTIAL | NO | conditional gate (E) built in cast.py; A-D not yet implemented |
| Baselines (NoSteer/SystemPrompt/FewShot/Unconditional/RefusalDir/CAST/Router) | YES (baselines.py) | YES | NO | uniform Baseline.respond interface; real runs PENDING |
| End-to-end safety eval driver (method vs baselines, Pareto, rigor) | YES (run_safety_eval.py) | YES (dry-run, 5 tests) | NO | injectable run() + --dry-run offline; real run PENDING |
| Gram-Schmidt orthogonalization (E19) | YES (multi_intent.py) | YES | NO | built |
| n>=7 seed evaluation (any axis) | NO | NO | NO | all evaluations are n=1 to n=20 synthetic or n=500 concepts under weak judge; method ladder not yet run |
| Statistical rigor (paired Wilcoxon + bootstrap CI + Holm-Bonferroni + ordinal gate) | YES | YES | PARTIAL | stats.py; Holm-vacuous FIXED (external_ready requires holm_applied); sign-aware verdict() added |

---

## External-ready findings

**Zero.** No result meets the full six-part rigor contract (n>=7 iid seeds, paired
Wilcoxon p<0.05, bootstrap CI excluding zero, Holm-Bonferroni correction, ordinal
gate, logged at rung 3+ in EXPERIMENT_LEDGER.md). All S-1..S-21 observations are
SCREENING ONLY and cannot be cited as research findings.

---

## What remains before the method can be evaluated (RUNS, not code)

The code for all of the below is BUILT and unit-tested (Waves A/B/C). The remaining
work is execution, blocked on machine resources + external data:

1. **Free VRAM / RAM** — Qwen-7B judge load previously OOM'd (Windows paging-file
   error 1455 at 109/129 GB committed). Restart / close memory hogs before running.
2. **Verify the safety-benchmark HF dataset ids** flagged `[VERIFY]` in
   src/steering/safety_bench.py against the live Hub (JBB id is confirmed-style;
   StrongREJECT/HarmBench/AdvBench/XSTest/SORRY-Bench are best-known mirrors).
3. **Calibrate the Qwen-7B judge** (safety_judge.calibrate) against each
   benchmark's own labels; do not trust compliance numbers until agreement is high
   (the behavior judge is currently AUC 0.68 — below the 0.85 bar).
4. **Climb the method ladder** (docs/METHOD_LADDER.md): refusal-dir -> in-forward
   conditional gate -> multi-intent -> over-refusal control -> adversarial -> SOTA.
5. **Pre-register** the success criterion in git BEFORE the evaluation sweep.
6. **Run** `scripts/run_safety_eval.py` (dry-run smoke first) and the adversarial
   red-team; report Pareto (ASR vs over-refusal) at n>=7 with the rigor contract.

Smoke now (offline, no GPU): `PYTHONPATH=src python scripts/run_safety_eval.py --dry-run --no-log`.

---

## Reviewer status

| Reviewer | Verdict | Score | Key complaint |
|---|---|---|---|
| ICLR | Reject | 2/10 | Method does not exist; safety axis is a regex |
| ICML | Conditional accept (blocker: lint/type) | -- | Blocker fixed; methodology sound |
| NeurIPS | Reject | 2/10 | Judge AUC 0.68; zero external-ready; prompting baseline missing |
| Elite researcher | Reject | 2/10 | The one rigorous result is negative; no SOTA claim possible |
| Lab leader | Reject | 2/10 | No multi-intent definition; no real safety benchmark wired |

Roadmap: audits/reviews/IMPROVEMENTS_100.md (100 items). Build status: groups A
(method), B (benchmarks+judge), C (multi-intent def), D (baselines), E (rigor
fixes), F (adversarial), H (transparency), I (ARCHITECTURE/CI/reproduce), J
(anti-slop), L (paper) — BUILT (Waves A/B/C). Remaining: G (scale) and the
evaluation RUNS across A-F, which are deferred until VRAM is freed.
