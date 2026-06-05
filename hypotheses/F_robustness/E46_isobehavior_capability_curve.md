# E46 — Iso-Behavior Capability Curve: Separately Tunable Efficacy and Tax

> **One-line claim:** Efficacy (behavior-success rate) and capability tax
> (MMLU drop) are separately tunable: certain (layer, alpha, sparsity)
> combinations hold behavior fixed while sweeping capability tax along an
> iso-behavior contour, demonstrating that the behavior-capability trade-
> off is not a single monotone curve but a 2D surface with iso-behavior
> ridges.
>
> **Block:** F — Robustness, safety, and evaluation (E41-E50).
> **Primary axis:** A3 (HOW MUCH — coefficient).
> **Implementation status:** `o planned / UNTESTED`.

---

## In Plain English

**What we're testing, simply:** Usually we assume "more steering = more behavior
but more damage to the model's smarts," as one fixed trade. We ask whether you
can actually hold the behavior steady while separately dialing the damage up or
down — meaning the trade-off is two knobs, not one.

**Key terms (defined here):**
- **Language model** — an AI that writes text one word at a time.
- **Steering** — changing the model's behavior by editing its internal state
  mid-sentence, without retraining.
- **Steering vector** — the nudge we add to push toward a behavior.
- **Residual stream** — the model's running internal scratchpad; the nudge goes
  here.
- **Layer** — one of the model's stacked processing steps; *which* step is one
  of the knobs.
- **alpha / strength** — how hard we push; another knob.
- **DiffMean** — the simplest nudge recipe: average internal state on "yes"
  examples minus "no" examples. No training.
- **Sparsity** — how few of the nudge's internal numbers we keep (a third knob).
- **Efficacy** — how strongly the target behavior shows up in the output.
- **Capability tax** — how much the model's general smarts drop as a side effect
  (measured by whether it still answers quiz questions).
- **Iso-behavior contour** — a set of knob settings that all give the *same*
  behavior level. Moving along it changes only the damage, not the behavior.
- **Coherence** — whether the steered text stays fluent and sensible.

**Why we're doing this (the point):** If behavior and damage can be tuned
separately, we could get the behavior we want at the *lowest possible* cost to
the model's intelligence — instead of being forced to accept whatever damage
comes with it.

**What the result would mean:** If we can hold behavior fixed while shrinking the
damage, steering becomes far more controllable (a real practical gain). If
behavior and damage always move together, the trade-off really is a single fixed
curve and you can't cheat it.

See [`../GLOSSARY.md`](../GLOSSARY.md) for any other term.

---

## 1. Motivation (>= 100 words)

A common implicit assumption in steering experiments is that the trade-off
between behavior efficacy and capability cost is a single monotone curve:
more behavior requires more alpha, which causes more capability loss,
with no way to have the same behavior at different capability costs. This
assumption would be true if the residual stream were a uniform space where
every dimension equally affects both behavior and capability. However, the
multi-axis steering framework (IDEA_TABLE.md, 12-axis framework) suggests
that the (layer, alpha, sparsity, operation) parameter space is much richer:
different combinations of these parameters can achieve the same behavior
efficacy via different paths through the activation manifold, and those
different paths may have different capability costs. The N5/N17 geometry
provides a specific prediction: what matters for capability cost is the off-
shell displacement (how far h is pushed off the data manifold), and what
matters for behavior efficacy is the projection onto the behavior direction.
These two quantities can be partially decoupled: a sparse vector that injects
only along the behavior-relevant coordinates may achieve the same behavior
projection (efficacy) as a dense vector at lower total displacement
(capability cost). The iso-behavior contour — the set of (layer, alpha,
sparsity) settings that achieve the same behavior-success rate — is a 2D
surface in parameter space, and along this surface the capability cost
varies. E46 maps this surface and identifies the minimum-capability-cost
operating point. This connects directly to N17 (off-shell displacement
governs incoherence), N5 (norm-budget conservation), and the N15/N16 geometry
(angle vs radius decoupling: the behaviour is in the angle, the capability
cost is in the radius). The Rogue Scalpel Guard Layer B (manifold norm clamp)
is the operational form of the iso-behavior contour: it holds the radius
fixed while allowing the angle (behavior) to move.

---

## 2. Formal Hypothesis (>= 50 words)

**H:** On Gemma-2-2B-it, for the refusal behavior direction, there exist
at least two distinct (layer, alpha, sparsity) parameter combinations that
achieve the same behavior-success rate (within +-5 percentage points) but
differ in MMLU capability drop by >= 3 percentage points, demonstrating
that iso-behavior settings with different capability costs exist. The lower-
capability-cost setting is predicted to correspond to a lower off-shell
displacement ||delta h|| / ||h|| at matched behavior projection, consistent
with the N17 geometry.

---

## 3. Falsifier (>= 30 words)

If no two (layer, alpha, sparsity) combinations achieve the same behavior-
success rate (within +-5 pp) but differ in MMLU drop by >= 2 pp, the iso-
behavior surface is flat in capability and the separate tunability claim is
DISCARDED (Status `x disproved`). A flat surface means the behavior-capability
trade-off is irreducibly 1D.

---

## 4. Citations (Citation-Rigor format, >= 80 words)

```
Korznikov, et al. 2025 'The Rogue Scalpel: Activation Steering Compromises
LLM Safety' arXiv:2509.22067 — Rogue Scalpel; Guard Layer B (norm clamp)
is the operational realisation of the iso-behavior contour: clamp ||delta h||
while maintaining the behavior direction, decoupling off-shell displacement
from behavior efficacy; this experiment tests whether that decoupling works
empirically.

Project screening results: FINDINGS.md S-6 (N17: log PPL = 5.40 + 2.87 *
offshell, R^2=0.81) and S-9 (C9b: clean steering window at 10% off-shell
displacement) establish that off-shell displacement governs capability cost
independently of steering direction; these are the screening results that
motivate the iso-behavior tunability claim.

Yin, et al. 2024 'Selective Activation Steering' arXiv:2601.19375
(Selective) — sparse per-head steering; its selective injection achieves
comparable behavior at lower total displacement than full-residual injection;
an implicit iso-behavior-lower-cost design that E46 explicitly characterises.

Zou, Andy, et al. 2023 'Representation Engineering: A Top-Down Approach
to AI Transparency' arXiv:2310.01405 — CAA / DiffMean baseline; the full-
residual injection method that E46 compares against sparse/layer-selected
alternatives on the iso-behavior contour.
```

---

## 5. Mechanism

The iso-behavior contour is defined by fixing the behavior-success rate and
varying the steering parameters. The key degrees of freedom are:

1. **Layer**: later layers have different manifold curvature; the same
   behavior projection may be achievable at a later layer with lower
   off-shell displacement (because the manifold is "flatter" near the
   final layers where the model is more directly predicting outputs).
2. **Alpha**: the magnitude of the injection; at a later or more behavior-
   relevant layer, a lower alpha achieves the same projection.
3. **Sparsity**: a sparse vector (top-k coordinates) achieves the same
   behavior projection if those coordinates carry the behavior signal, but
   with lower total ||delta h|| because the small-magnitude coordinates
   (which contribute to capability cost via off-manifold displacement) are
   zeroed out.

The iso-behavior curve in (layer, alpha) space is approximately:

    behavior(layer, alpha) ~ C * alpha * projection_efficiency(layer)

where projection_efficiency(layer) is how much of a unit alpha injection
at layer l contributes to the behavior direction. Keeping behavior constant
while minimising off-shell displacement means choosing the (layer, alpha)
point on the iso-behavior curve with the lowest off-shell displacement —
which, by N17, predicts the lowest capability cost.

The N16 CRH connection: behavior is in the angle (direction), capability
cost is in the radius (norm). Iso-behavior = same angle change. Minimum
capability cost = minimum radius change. The iso-behavior surface is the
surface of constant angle at varying radius — directly analogous to N16's
angle/radius decoupling (R^2=0.997 screening support for angular prediction
of rotation PPL, S-7).

---

## 6. Predicted Delta

| (Layer, alpha, sparsity) setting | Behavior success | MMLU drop |
|---|---|---|
| Early layer, high alpha, dense | 75% | -5 pp |
| Late layer, low alpha, dense | 75% | -2 pp |
| Optimal layer, low alpha, sparse (10%) | 75% | -1 pp |
| Predicted iso-behavior minimum-cost | 75% | -1 to -2 pp |

Key prediction: at matched behavior (75%), the minimum-cost setting
achieves MMLU drop 3-4 pp lower than the baseline (early/dense) setting.
The off-shell displacement at the minimum-cost setting will be 30-50%
lower than at the baseline setting.

---

## 7. Protocol

### 7.1 Primary experiment

- Model: Gemma-2-2B-it (4-bit), RTX 4090.
- Behavior: refusal (primary); sycophancy (secondary).
- Parameter sweep: 3 layers (early: L8, middle: L16, late: L22) x 3 alpha
  values (0.05, 0.10, 0.20 relative_add) x 3 sparsity levels (100%, 25%,
  10% of d_model) = 27 conditions per behavior.
- Eval: behavior-success rate (LLM-judge), MMLU-500 delta, WikiText-103 PPL,
  off-shell displacement ||delta h|| / ||h||.
- Iso-behavior identification: group conditions by behavior-success rate
  (within +-5 pp bins); within each bin, identify the range of MMLU drops.
- Geometry check: within each iso-behavior bin, verify that MMLU drop
  correlates with off-shell displacement (N17 prediction).
- Seeds: 3 (screening), 7 for rung-3.

### 7.2 Where it shines

This experiment transforms the alpha-tuning question from a 1D search to
a multi-dimensional optimisation with a principled target (iso-behavior,
minimum capability cost). The practical output is a recommended operating
point for each behavior that minimises capability tax.

---

## 8. Cross-references

- IDEA_TABLE.md Block F row E46.
- E44 (safety-capability Pareto frontier): E44 traces the frontier across
  behavior levels; E46 traces the iso-behavior contour across capability
  costs. Together they characterise the full 2D surface.
- N17 (off-shell displacement predicts incoherence): FINDINGS.md S-6
  (R^2=0.81, screening) is the key prior; E46 tests whether off-shell
  also predicts MMLU drop at matched behavior (not just PPL).
- N16 (CRH, angle/radius decoupling): screening support S-7 (R^2=0.997
  for angular prediction of rotation PPL); the iso-behavior contour is
  the constant-angle surface.
- N5 (norm-budget conservation): the iso-behavior minimum-cost point is
  the point of minimum off-shell displacement on the iso-behavior contour.
- Rogue Scalpel Guard B (norm clamp): E46 tests whether Guard B's norm
  clamping reduces MMLU drop at matched behavior.

---

## 9. Committee Q&A

**Q: Isn't "iso-behavior" hard to measure precisely? Behavior-success rates
have wide confidence intervals at n=3.**

> The +-5 pp iso-behavior bin is a practical tolerance, not an exact iso-
> behavior constraint. Within a +-5 pp bin, the MMLU drop range should be
> identifiable if the range itself is >= 3 pp (the falsifier threshold).
> At n=7 seeds, the CI on behavior-success rate is approximately +-10 pp
> at a 70% rate; the +-5 pp iso-behavior bin requires n >= 7 to be reliable.
> This is why rung-3 (n >= 7) is needed for the definitive result.

**Q: Could the capability-cost difference within an iso-behavior bin be
explained by different injection layers having different coherence properties
(not behavior-direction geometry)?**

> Yes; this is a confound. The geometry check (off-shell displacement within
> each bin) is the diagnostic: if off-shell displacement predicts MMLU drop
> within the iso-behavior bin (consistent with N17), the geometry story
> holds. If not, layer-specific coherence is the explanation.

---

## 10. Verification checklist

- [ ] 27-condition parameter sweep pre-registered before running.
- [ ] Off-shell displacement computed at each condition.
- [ ] Iso-behavior bin tolerance (+-5 pp) pre-registered.
- [ ] MMLU drop and off-shell displacement correlation computed within each
      bin (N17 geometry check).
- [ ] Rung-3 gate: n >= 7, Wilcoxon + bootstrap CI (Holm corrected).
- [ ] IDEA_TABLE.md row updated.
- [ ] N17 screening cross-reference (FINDINGS.md S-6) noted in results.

---

## 11. Status Journal

- 2026-05-31 — Created from corpus block F, hypothesis E46.
  Status: `o UNTESTED`.

  Cross-ref: FINDINGS.md S-6 (N17 law: logPPL = 5.40 + 2.87 * offshell,
  R^2=0.81) and S-7 (N16: angular predicts rotation PPL R^2=0.997) are
  the screening geometry results that motivate the iso-behavior decoupling
  claim. S-9 (C9b: behavior peak at 10% off-shell) establishes the clean
  steering window. E46's central claim (same behavior, different capability
  cost, predicted by off-shell geometry) is the multi-parameter extension
  of these screening findings. No prior direct screening of E46 itself.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-F (multi-objective optimisation + geometry specialist).*

### Prior plausibility
**MEDIUM-HIGH.** The N17 geometry result (R^2=0.81 screening) strongly
supports the idea that off-shell displacement is the key variable for
capability cost. The iso-behavior separation claim follows directly if
off-shell can be controlled independently of behavior projection. The C9b
relative_add result (behavior peaks at ~10% displacement) suggests the
iso-behavior contour is non-degenerate.

### Mechanism scrutiny
The mechanism is well-grounded in the N17/N16 geometry. The main risk is
that "iso-behavior" in terms of behavior-success rate may not correspond
to iso-behavior in terms of the actual behavior direction projection. If the
LLM-judge measures surface features of the output rather than the
underlying activation projection, the "iso-behavior" bin may contain
conditions that are mechanistically different.

### Confounds
1. MMLU-500 measures a specific set of knowledge questions; it may not
   capture all capability dimensions. A broader capability measure (ARC,
   GSM8K, HellaSwag) would strengthen the claim.
2. The 27-condition sweep may not have fine enough resolution to detect
   a 3 pp MMLU difference within a +-5 pp behavior bin; statistical power
   analysis is recommended before running.

### Expected effect size
My prior: within an iso-behavior bin, MMLU drop range of 2-5 pp (straddling
the 3 pp falsifier). The separation is likely real but the effect size may
be smaller than predicted. Recommend running the sweep at n=7 from the
start.

### Verdict
**TESTABLE + GEOMETRY-GROUNDED** — The iso-behavior contour mapping is a
theoretically motivated and practically useful experiment. It directly
tests the decoupling of behavior and capability in the multi-parameter
steering space and provides the geometry-check for N17 in the multi-
dimensional setting.

---

## Pseudocode & Methodology

This section specializes [`../METHODOLOGY.md`](../METHODOLOGY.md) to E46, a
**robustness/geometry** test: are efficacy and capability tax separately tunable?
A (layer x alpha x sparsity) sweep should produce iso-behavior settings with
different MMLU costs, predicted by off-shell displacement.

### 1. Steering-vector recipe (DiffMean + optional sparsity mask)

```python
# §1.3 METHODOLOGY: DiffMean refusal direction, unit-normalised.
v = bank("refusal")["diffmean"]; v = v / norm(v)
# sparsity level s in {1.00, 0.25, 0.10}: top-k coordinate mask (as in E35).
v_s = topk_coordinate_mask(v, frac=s)            # zero all but the largest-|.| s*d_model coords
```

### 2. Experiment procedure (27-cell sweep, iso-behavior bins)

```text
1. 3 layers (early L8, middle L16, late L22) x 3 alphas (0.05,0.10,0.20
   relative_add) x 3 sparsity (1.00,0.25,0.10) = 27 conditions per behavior.
2. Inject each via hooks.apply_operation, operation="relative_add" (§2).
3. MEASURE (§3 METHODOLOGY) per condition: behavior efficacy (off-family judge);
   MMLU-500 delta; WikiText PPL; off-shell ||Δh||/||h|| (geometry probe).
4. Group conditions into iso-behavior bins (behavior-success within ±5 pp);
   within each bin, report the RANGE of MMLU drops and correlate MMLU drop with
   off-shell displacement (the N17 check).
```

### 3. Measurement & decision rule

- **PRIMARY metric:** within an iso-behavior bin (±5 pp), the spread of MMLU drop
  across (layer, alpha, sparsity) settings, and whether off-shell predicts it.
- **Hypothesis (§2):** >= two settings with same behavior (±5 pp) differ in MMLU
  drop by >= 3 pp; the lower-cost setting has lower off-shell displacement (N17).
- **Pre-registered FALSIFIER (§3):** if NO two settings match in behavior (±5 pp)
  yet differ in MMLU drop by >= 2 pp, the iso-behavior surface is flat in
  capability and separate tunability is DISCARDED (`x disproved`).

### 4. Where the code is / status — UNTESTED (geometry screened)

- **No driver yet** (campaign + `scripts/build_provenance.py` -> `PROVENANCE/E46.md`).
- **Screening anchors:** FINDINGS.md S-6 (N17 R^2=0.81) and S-7 (N16 angular,
  R^2=0.997) motivate the off-shell decoupling; S-9 sets the clean window.
  **Missing machinery (why UNTESTED):** the **top-k coordinate mask** sweep (E35
  dependency); the **27-cell layer x alpha x sparsity driver** with per-cell MMLU
  + off-shell logging; generation-judge efficacy on Gemma-2-2B at n>=7 for the ±5 pp
  bins to be reliable. The geometry probes exist; the full surface sweep does not.

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/E46.md`.
