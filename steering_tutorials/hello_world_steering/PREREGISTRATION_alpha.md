# Pre-registration — extending the alpha grid to 0.20 and 0.25

**Written 2026-08-01, BEFORE the run. Not revised afterwards.**
The post-hoc result lives in `README.md` and `artifacts/results.json`; this file
is frozen so the prediction can be graded against what actually happened.

---

## 1. The coverage gap being closed

This lesson sweeps `alpha in [0.0, 0.05, 0.10, 0.15]` and concludes that a
diff-of-means vector does **not** install refusal in the abliterated
Gemma-3-1B: on the held-out harmful half, refusal falls
`0.330 -> 0.220 -> 0.120 -> 0.040` while gibberish climbs
`0.225 -> 0.410 -> 0.630 -> 0.845` (base-extracted direction, n=200/cell,
off-family Qwen2.5-3B judge).

The sibling lesson `steering_tutorials/realignment` uses the **same** steered
model, the **same** layer 12, the **same** `relative_add` hook, and a direction
that is cosine **0.99999988** to this lesson's base-extracted vector — and its
best operating point is **alpha = 0.25**, which is *outside* this lesson's grid.

So this lesson has never measured the alpha at which its own sibling
demonstrably works. That is a **coverage hole**, not a tuning opportunity.

## 2. The ONE variable that moves

The alpha grid, from `[0.0, 0.05, 0.10, 0.15]` to
`[0.0, 0.05, 0.10, 0.15, 0.20, 0.25]` — a strict **superset**. Every existing
alpha is kept and will be reported.

**Held fixed** (any change here would confound the comparison):
judge (`STEER_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct`), the deterministic
repetition/gibberish gate and its `_DISTINCT_RATIO_FLOOR=0.35`, `N_PER_CLASS=500`
/ `N_EXTRACT=300` / n=200 held-out harmful per cell, the prompt set and its
seed-0 split, `STEER_LAYER=12`, `MAX_NEW_TOKENS=48`, greedy decoding, and both
extraction sources (`STEER_EXTRACT_FROM=both`: aligned base = primary, and the
abliterated same-model ablation). The four already-measured alphas are reused
from the run checkpoint, not re-measured — the cells are independent under
greedy decoding, and the checkpoint carries the config fingerprint that proves
they came from this exact configuration.

**No "best alpha" will be selected or introduced.** The conditional arm's alpha
continues to be chosen by the pre-existing `choose_conditional_alpha` rule,
unmodified.

## 3. Prediction (registered before the run)

**P1 — refusal does NOT rise.** On the base-extracted (primary) curve,
`refusal_rate(0.20)` and `refusal_rate(0.25)` will both be **<= 0.12** (the value
already measured at alpha=0.10), and neither will exceed `refusal_rate(0.05)
= 0.220`. Mechanistically: what the previous sweep shows is monotone **removal of
compliance** (0.445 -> 0.115), with the freed mass going almost entirely to
GIBBERISH rather than to REFUSAL. If the vector installed a refusal behaviour,
refusal would have to grow somewhere on the dose-response curve; instead every
increment of alpha buys incoherence. Pushing alpha further should push further
along that same trajectory.

**P2 — gibberish keeps climbing.** `gibberish_rate(0.25) >= gibberish_rate(0.15)
= 0.845`, and I expect it to approach ~1.0 (point estimate 0.93-1.00).

**P3 — compliance keeps falling.** `compliance_rate(0.25) <= 0.115`, plausibly
near 0.

**P4 — the non-compliant split stays gibberish-dominated.** Of the outputs that
are not COMPLIANCE, the REFUSAL share
`refusal / (refusal + gibberish)` will be **below 0.15** at alpha 0.20 and 0.25
(it is 0.595 at alpha=0, 0.349 at 0.05, 0.160 at 0.10, 0.045 at 0.15). This is
the sharpest form of the claim: the vector does not convert compliance into
refusal, it converts compliance into noise.

**P5 — the abliterated-extracted ablation behaves the same way.** Its refusal at
0.20 and 0.25 will not exceed its own maximum so far (0.21 at alpha=0.10).

## 4. Falsifier — what would prove me wrong

If `refusal_rate` at alpha 0.20 or 0.25 **exceeds 0.330** (the unsteered
baseline) on either extraction source, then steering *does* install refusal at a
dose this lesson simply never tested. That is a genuine reproduction of the
CAA (arXiv:2312.06681) / ActAdd (arXiv:2308.10248) claim and **must be reported
as one**, with the paper verdicts revised upward — regardless of what this
document predicted. A weaker but still real positive would be any rise above
`refusal_rate(0.05) = 0.220` while `gibberish_rate` stays below 0.845.

## 5. What each outcome means for the paper verdicts

- **Refusal rises above baseline at 0.20/0.25** -> ActAdd / CAA reproduced on the
  behaviour-installation claim; the previous "not supported" and the current
  "compliance-removal only" readings were both grid-limited artifacts.
- **Refusal never rises, gibberish -> ~1.0** -> the honest conclusion is stated
  without softening: *this vector removes compliance and destroys coherence
  without installing refusal*, at every dose tested up to and including the dose
  at which the sibling lesson's operating point sits. The Arditi
  (arXiv:2406.11717) directional-ablation reading (a single mid-layer direction
  mediates refusal-vs-compliance) survives on the compliance side; the CAA /
  ActAdd claim that *adding* the direction installs the target behaviour does
  not, in this configuration.

## 6. Registered analysis, fixed in advance

Report, for **both** extraction sources and **all six** alphas: refusal,
compliance, gibberish, n, and the derived non-compliant split
`refusal / (refusal + gibberish)`. No alpha is dropped, highlighted as "best",
or de-emphasised. The non-compliant split is pure arithmetic on the measured
rates — it introduces no new measurement and is computed identically for the
already-measured alphas.
