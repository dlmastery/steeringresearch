# M8 — Conformal Conditional Gate with a Provable Over-Refusal Guarantee

> **One-line claim:** A split-conformal calibration of the per-intent cosine gate
> gives a distribution-free, finite-sample guarantee that the benign over-refusal
> rate is <= alpha (e.g. 1%), while retaining high harmful recall — turning the
> gate from a "moderate AUC 0.74 black box" into a *certified* selectivity control
> that Pareto-dominates an uncalibrated steerer and a classifier-router.
>
> **Source design space:** Block B — Conditional / Gated Steering; Block G method
> component (M-series). This is item #9 of the lab-leader strategic review ("bet on
> the gate, not the vector").
>
> **Implementation status:** `DESIGN + PLAN — RUN PENDING`. The cosine gate exists
> (`gate.CosineGate`, `intent_gate.calibrate_thresholds`); the conformal wrapper is
> NOT yet implemented. This doc is a pre-registered plan, not a result. Gated behind
> a hard precondition: a judge calibrated to AUC >= 0.85 (see §7.0).

---

## In Plain English

**What we're testing, simply:** The gate decides "is this request the kind we should
steer toward refusal?" by checking how harmful-looking the request's internal state is
against a threshold. The question here is not "is the gate accurate?" (we already know
it's only moderate) but: **can we set the threshold so that we can PROMISE, with a
math guarantee, that the gate wrongly trips on at most 1% of harmless requests?** —
and still catch most harmful ones. A promise like "provably <= 1% over-refusal" is
something a product team can actually ship.

**Key terms:**
- **Gate / cosine threshold:** steer only when cos(internal state, harm-direction)
  exceeds a cut-off tau.
- **Over-refusal (false positive):** the gate trips on a *benign* request, so a safe
  request gets needlessly refused. The thing users hate.
- **Harmful recall (true positive):** the gate trips on a genuinely harmful request.
- **Conformal prediction:** a standard, distribution-free statistical method that
  picks the threshold from a held-out *calibration* set so a future error rate is
  provably bounded — no assumption about the model or data shape, only that
  calibration and future prompts are drawn alike (exchangeable).
- **Held-out calibration:** the threshold is chosen on data NOT used to score the
  result — which fixes the S-25 confound where tau was set in-sample.

**Why we're doing this:** Every prior gate result here reports an AUC (a quality
score) but no *operating guarantee*. Industry safety systems live or die on the
over-refusal rate. A gate you can certify ("<= 1% over-refusal under this
distribution") is a genuinely novel, ownable contribution — more defensible than
chasing a fragile AUC improvement that S-26 showed a trained probe can't even deliver.

**What the result would mean:** SUPPORTED ⇒ we own "certified-selectivity conditional
steering". FALSIFIED (coverage breaks under realistic prompt shift) ⇒ an equally
publishable negative: conformal gating's exchangeability assumption does not survive
jailbreak-style distribution shift, quantifying exactly how much.

See [`../GLOSSARY.md`](../GLOSSARY.md).

---

## 1. Motivation (>= 100 words)

The repo's positive signal is entirely gate-side: the DiffMean direction detects
concept-relevant inputs at mean AUC 0.747 (S-25), the raw cosine gate beats a trained
logistic probe that overfits to held-out AUC 0.43 (S-26, E15), and conditional gating
preserves off-target fluency that unconditional steering destroys (S-25, withdrawn as
a number because tau was set in-sample). Three independent reviews converge: the gate,
not the steering direction (E7 NULL), is the contribution. But a moderate AUC is not a
deployable artifact — a deployer needs a *guaranteed operating point*, specifically a
bounded over-refusal rate, because over-refusal is the dominant production complaint
for safety-tuned models (and gemma-2-2b already over-refuses ~33% of XSTest benign
prompts, S-22). Split-conformal prediction converts any score function (here the gate
cosine) into a threshold with a finite-sample, distribution-free guarantee on the
false-positive (benign-firing) rate. Applying it per intent yields a gate that
certifies "<= alpha over-refusal" without retraining, without an accurate probe, and
without the in-sample-tau confound that invalidated S-25.

---

## 2. Formal Hypothesis (>= 50 words)

Because split-conformal calibration sets the gate threshold tau as the upper
empirical (1 - alpha) quantile of benign-calibration cosine scores, and because the
conformal guarantee P(score_new > tau | benign) <= alpha holds for any exchangeable
benign distribution regardless of the score function's accuracy, the conformally
gated steerer will (a) realize a benign over-refusal rate <= alpha + 1/(n_cal + 1) on
a held-out in-distribution benign test set, and (b) at alpha = 0.01 retain harmful
recall >= 0.70 on a held-out harmful set, thereby Pareto-dominating BOTH the
unconditional steerer (which fires on 100% of benign, no FPR control) AND a
token-space classifier-router at matched recall, on the (over-refusal, recall,
latency) frontier. The mechanism is finite-sample quantile coverage, not gate
accuracy: even a moderate AUC-0.74 score function yields a *valid* (if not tight)
selectivity certificate.

---

## 3. Falsifier (>= 30 words)

FALSIFIED if, on a held-out IN-DISTRIBUTION benign test set, the realized over-refusal
rate exceeds alpha + 0.02 (coverage violation ⇒ exchangeability or implementation
broken). Separately, the OPERATIONAL claim is FALSIFIED if at the conformal alpha=0.01
operating point harmful recall < 0.70, OR if the conformal gate does not Pareto-
dominate the classifier-router on at least one of {over-refusal at matched recall,
latency, paraphrase-attack robustness}. Distribution-shift coverage (jailbreak-style
benign-looking harmful prompts) is reported separately, not as the primary falsifier
(conformal makes no cross-distribution promise).

---

## 4. Citations (Citation Rigor >= 80 words)

```
Vovk, V., Gammerman, A., Shafer, G. 2005 Springer 'Algorithmic Learning in a
Random World' — split/inductive conformal prediction; the finite-sample marginal
coverage guarantee P(score_new > tau) <= alpha from the (1-alpha) calibration
quantile is the core mechanism transplanted here onto the gate cosine.

Angelopoulos, A. N., Bates, S. 2023 FnTML 'Conformal Prediction: A Gentle
Introduction' (arXiv:2107.07511) — the modern split-conformal recipe and the
n+1 finite-sample correction tau = ceil((n+1)(1-alpha))-th order statistic; the
exchangeability assumption and its failure under distribution shift (the M8 §3
distribution-shift caveat).

Wu, Yuming, et al. 2024 arXiv 'Conditional Activation Steering' (arXiv:2409.05907)
— CAST fixed cosine gate; M8 adds the calibration layer CAST lacks (CAST picks
theta_c on a dev set with no coverage guarantee).

Arditi, Andy, et al. 2024 arXiv 'Refusal in LMs Is Mediated by a Single Direction'
(arXiv:2406.11717) — the refusal/harm direction whose cosine is the conformal score
function; M1 extracts it from real harmful-vs-benign PROMPTS (the input-triggered
signal the gate needs, unlike AxBench concept induction, S-25).

Rottger, P., et al. 2024 NAACL 'XSTest: A Test Suite for Identifying Exaggerated
Safety Behaviours' (arXiv:2308.01263) — the benign over-refusal evaluation set; its
250 safe prompts are the conformal calibration + test pool.
```

---

## 5. Mechanism

### 5.1 Split-conformal threshold (per intent)

Score function s(x) = cos(pool(h@L_c)(x), v_intent), where v_intent is the M1 refusal
direction for that intent (extracted from harmful-vs-benign PROMPTS, not AxBench
concepts). Given a CALIBRATION set of n benign prompts with scores s_1..s_n:

    tau_alpha = the k-th LARGEST calibration score, k = ceil((n+1) * alpha)

Then for an exchangeable future benign prompt, P(s_new > tau_alpha) <= alpha
(distribution-free, finite-sample). The gate fires (steers) iff s(x) > tau_alpha.
Calibration set is DISJOINT from the evaluation set (fixes the S-25 in-sample-tau
confound). No training, no probe (S-26 says the probe overfits anyway).

### 5.2 Why a weak score still gives a valid certificate

Coverage depends only on the rank of s_new among calibration scores under
exchangeability — NOT on whether s separates classes well. A moderate AUC-0.74 score
yields a VALID certificate; better separation only makes it TIGHTER (higher recall at
the same alpha). This decouples "the guarantee" (always valid) from "the usefulness"
(recall at alpha), which is the honest, publishable framing.

### 5.3 Multi-intent

One tau_alpha per intent (Bonferroni-split alpha across K intents for a joint
guarantee, or per-intent marginal). Compose fired intents via `multi_intent`
(Gram-Schmidt) as in M3.

### 5.4 Baselines it must beat (item #6 of the review)

- **Unconditional steerer:** fires always ⇒ benign over-refusal ~100% bounded only by
  the steer's own damage. Trivially dominated on over-refusal.
- **Classifier-router (the production baseline):** a token-space classifier (e.g. the
  Qwen judge used as a router, or Llama-Guard-style) decides refuse/answer. M8 must
  win on >= 1 of: over-refusal at matched recall, latency (one forward pass vs two
  models), or paraphrase-attack robustness (activation-space vs token-space).
- **Prompting / system-prompt refusal.**

---

## 6. Predicted Delta

| Metric | Predicted | Rationale |
|---|---|---|
| Realized benign over-refusal at conformal alpha=0.01 | [0.00, 0.03] | coverage = alpha + 1/(n_cal+1); n_cal~125 ⇒ <= ~0.018 |
| Harmful recall at alpha=0.01 | [0.55, 0.80] | from AUC~0.74 score; the usefulness number, NOT the guarantee |
| Coverage violation (held-out in-dist) | False | conformal validity if exchangeable + correct impl |
| Coverage under jailbreak shift | likely VIOLATED | reported as the distribution-shift finding, not a failure |
| Pareto vs unconditional (over-refusal axis) | dominates | unconditional fires on all benign |
| Latency vs classifier-router | lower | single forward pass, no second model |

Pre-registered; [NEEDS VERIFICATION]. All numbers are predictions, not results.

---

## 7. Experimental Protocol

### 7.0 HARD PRECONDITION (gate before any M8 run)

**The judge must be calibrated to AUC >= 0.85** against ground-truth refuse/comply
labels (review item #1; PREREGISTRATION §6 + Amendment 1). The current behavior judge
is AUC 0.68 and the Qwen safety judge is 9-10/12 dev with a conservative bias — both
below bar. Until that clears, M8 runs are SCREENING ONLY and cannot ground the recall
number (the over-refusal *coverage* claim is judge-independent for "fired/not-fired",
but "recall" needs the harmful label). Pre-check (cheap, judge-free): detector ROC of
v_intent on real harmful-vs-benign PROMPTS; abort if < 0.85 (per research review).

### 7.1 Primary experiment

- **Model:** gemma-3-1b-it (dev; per the current standard config) → 9B scale check.
- **Direction:** M1 refusal direction from real harmful (JBB-harmful) vs benign
  (JBB-benign) PROMPTS — input-triggered, not AxBench concepts.
- **Calibration set:** 125 XSTest-safe prompts (held out). **Test set:** the other
  125 XSTest-safe + JBB-harmful (disjoint).
- **Procedure:** compute tau_alpha for alpha in {0.005, 0.01, 0.05}; gate-fire on the
  test set; measure realized benign over-refusal (coverage) and harmful recall.
- **Controls:** repeat with v = matched-norm random, shuffled-label, and orthogonalized
  direction (E7 control suite) — coverage must hold for ALL (it's score-agnostic);
  recall must be HIGHER for the real direction if the direction carries signal.
- **Seeds / units:** bootstrap over calibration-set resamples x prompt bootstrap (the
  redefined replication unit, PREREG Amendment 1c), n >= 7.
- **Wall-clock:** minutes (one direction extraction + cosine scores; no generation
  needed for the gate-coverage measurement; generation only for recall judging).

### 7.2 Where it shines

The coverage guarantee is the headline and is judge-independent. The recall and the
router comparison need the calibrated judge. Report coverage FIRST (clean, certain),
recall SECOND (judge-gated).

---

## 8. Cross-References

- **M1** (refusal direction) — provides v_intent; M8 blocked on M1.
- **M2** (gate fires harmful not benign) — M8 is M2 + a coverage certificate.
- **E15 / S-26** (trained gate overfits) — justifies using the raw cosine score, not a probe.
- **E12** (energy-ratio gate) — alternative score function; conformal wraps either.
- **N17** (displacement budget) — composable guard; orthogonal to the gate.
- **PREREGISTRATION.md** Amendment 1 (over-refusal endpoint), **IDEA_TABLE.md** Block G row M8.

---

## 9. Committee Q&A

**Q: A guarantee from a weak (AUC 0.74) score sounds too good.**
> The guarantee is on the FALSE-POSITIVE (over-refusal) rate only, and it is a
> ranking/quantile fact, independent of accuracy. A weak score gives a valid but LOOSE
> certificate — you pay in RECALL (you catch fewer harmful prompts at the same alpha),
> not in the over-refusal promise. That trade is exactly what we measure and report.

**Q: Conformal assumes exchangeability — jailbreaks violate it.**
> Correct, and that is a feature: we report in-distribution coverage as the primary
> (valid) result and the jailbreak-shift coverage drop as a separate, quantified
> finding. "How fast does the certificate degrade under adaptive prompt shift?" is
> itself a publishable safety result.

**Q: Why not just tune tau to 1% FPR on a dev set (what intent_gate already does)?**
> Dev-set tuning gives no finite-sample guarantee and silently breaks on the eval set
> (the S-25 in-sample-tau confound). Split-conformal's n+1 correction gives a provable
> marginal bound and forces the calibration/eval split that makes the number honest.

---

## 10. Verification Checklist

- [ ] Judge calibrated to AUC >= 0.85 (PRECONDITION) OR result labeled SCREENING.
- [ ] Detector ROC pre-check on real harmful-vs-benign prompts >= 0.85.
- [ ] Calibration set DISJOINT from test set (no in-sample tau).
- [ ] tau_alpha computed with the n+1 finite-sample correction.
- [ ] Realized benign over-refusal compared to alpha + 1/(n_cal+1).
- [ ] Coverage verified for real AND control directions (score-agnostic check).
- [ ] Recall at alpha reported with judge-AUC disclosed.
- [ ] Classifier-router baseline run on the same split (over-refusal, recall, latency).
- [ ] Bootstrap CI over calibration-resample x prompt-bootstrap, n >= 7.
- [ ] IDEA_TABLE.md Block G row M8 updated; PROVENANCE/M8.md created on first run.

---

## 11. Status Journal

- 2026-06-09 — Design doc created (item #9 of the lab-leader strategic review;
  convergent with the two dispatched code/research reviews). Status: DESIGN + PLAN,
  RUN PENDING. Blocked on: (a) M1 real refusal-direction extraction; (b) judge
  calibration to AUC >= 0.85 (the gating precondition for the recall number); (c) the
  conformal wrapper around `gate.CosineGate` (trivial once M1 + the split exist).
  Explicitly fixes the S-25 in-sample-tau confound via held-out calibration.

---

## Pseudocode & Methodology

```python
# M8: split-conformal gate. Score-agnostic coverage; uses the M1 refusal direction.
import numpy as np

def conformal_tau(benign_cal_scores: np.ndarray, alpha: float) -> float:
    n = len(benign_cal_scores)
    k = int(np.ceil((n + 1) * alpha))            # n+1 finite-sample correction
    k = min(max(k, 1), n)
    # tau = k-th LARGEST score => at most alpha fraction of future benign exceed it
    return float(np.sort(benign_cal_scores)[::-1][k - 1])

# fire(x) = cos(pool(h@L_c)(x), v_intent) > tau_alpha   # steer iff fired
# coverage check: mean(test_benign_scores > tau) <= alpha + 1/(n_cal+1)
# recall:        mean(test_harmful_scores > tau)        # usefulness, judge-gated downstream
```

Decision rule: PRIMARY = realized in-distribution benign over-refusal vs the conformal
bound (judge-independent, must hold). SECONDARY = harmful recall at alpha=0.01 (>= 0.70
falsifier) and Pareto vs classifier-router (needs judge AUC >= 0.85). See
[`../METHODOLOGY.md`](../METHODOLOGY.md).

---

## Provenance & Tracing

`DESIGN + PLAN — no experiments yet.` On first run, create `PROVENANCE/M8.md` with the
exp#, reproduce command, and artifact links. Reproduce (once M1 + judge precondition
land):

```bash
# PYTHONPATH=src python scripts/run_conformal_gate.py --model models/google/gemma-3-1b-it \
#   --judge-model Qwen/Qwen2.5-3B-Instruct --alpha 0.01 --calib xstest_safe --no-log   # TO BE WRITTEN
```
