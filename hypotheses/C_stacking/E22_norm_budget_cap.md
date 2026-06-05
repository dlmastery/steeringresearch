# E22 — Norm Budget Cap: Quantile-Bounded Multi-Vector Stacking

> **One-line claim:** Total steering budget is how far sum(alpha_i v_i) pushes
> h outside the in-distribution shell; capping ||delta h|| at the empirical
> activation-norm quantile prevents collapse.
>
> **Source design space:** Block C — Stacking and Multi-Vector Composition (E17–E26).
>
> **Implementation status:** `PENDING — UNTESTED`. Multi-vector injection and
> norm-budget capping code not yet built. Closely connected to N5 (SUPPORTED).

---

## In Plain English

**What we're testing, simply:** Every nudge pushes the model's "thoughts" a little
off their natural spot. Push too far and the text breaks. The idea: set a hard
limit on the total push, so no matter how many nudges we stack, we never shove the
text past the breaking point.

**Key terms (defined here):**
- **Steering / steering vector:** nudging the model by adding a direction to its
  internal "thoughts"; the direction is the steering vector.
- **Residual stream:** the model's running internal state we edit mid-sentence.
- **Layer:** the processing step where we make the edit.
- **Alpha / strength:** how hard we push.
- **DiffMean:** the simple recipe for building a nudge.
- **Coherence:** whether the text stays fluent and sensible.
- **Stacking:** using several nudges at once.
- **The activation shell (in-distribution shell):** healthy "thoughts" all sit at
  roughly one natural size, like points on the surface of a ball. Edits that push
  the thought off that surface tend to break the text.
- **Norm budget:** the total amount of push the text can absorb before it breaks.
  This experiment proposes a **cap** — clamp the combined push at a safe size
  (set from the model's own typical activation sizes) so a big stack can't
  overshoot.

**Why we're doing this (the point):** It's a safety brake for stacking. We want to
add as many behaviors as possible *and* keep the text readable — a cap lets us
stack freely without falling off the cliff.

**What the result would mean:** If capping the total push prevents the text from
collapsing while keeping the behaviors working, we have a simple, reliable guard
rail for multi-behavior steering. If capping kills the behaviors too, the limit is
too blunt and we'd need a subtler control.

See [`../GLOSSARY.md`](../GLOSSARY.md) for any other term.

---

## 1. Motivation (>= 100 words)

The norm-budget conservation law (N5, SUPPORTED: logPPL = 5.40 + 2.87 * offshell,
R² = 0.81, C2, 23 rows on Gemma-2-2B) establishes that perplexity degradation
is a near-linear function of off-shell displacement: the distance by which the
steered activation h_steered exceeds the in-distribution shell (the q-quantile
of ||h|| in natural activations). This law holds for SINGLE-VECTOR steering.
E22 extends it to MULTI-VECTOR stacking: the claim is that the SAME N5 collapse
curve governs multi-vector displacement, so all multi-vector experiments from
E17–E22 collapse onto one master curve when plotted against ||delta h|| / ||h||
(the normalized off-shell displacement). Practically, this means a universal
intervention: cap sum_i(alpha_i * v_i) so that ||sum_i alpha_i v_i|| <= B,
where B = q-quantile of ||h|| (the empirical activation-norm budget). By capping,
any combination of vectors — regardless of their number or individual alphas —
will avoid the N5 collapse region. This turns the multi-vector incoherence
problem from an N-dimensional optimization into a single scalar constraint.

---

## 2. Formal Hypothesis (>= 50 words)

Because N5 establishes that logPPL = 5.40 + 2.87 * (||delta h|| / ||h||) with
R² = 0.81 for single-vector steering on Gemma-2-2B, and because the norm-budget
law predicts that the displacement ||delta h|| is the unique predictor of
coherence collapse regardless of HOW that displacement is generated (one strong
vector or many weak vectors), re-plotting E17–E22 experiments against the
normalized total displacement ||sum alpha_i v_i|| / ||h|| should produce a
single master collapse curve consistent with the N5 parameters (slope ~ 2.87,
intercept ~ 5.40). Formally: (a) the master curve Spearman rho between
(||delta h|| / ||h||) and logPPL across all multi-vector conditions in E17–E22
is >= 0.75; (b) capping ||delta h|| at the 75th percentile of the natural
||h|| distribution prevents logPPL from exceeding the collapse threshold
(logPPL_cap < 6.5) in all tested multi-vector configurations.

---

## 3. Falsifier (>= 30 words)

If re-plotting all multi-vector data against (||delta h|| / ||h||) yields
Spearman rho < 0.75 (the master curve does not generalize from single-vector
to multi-vector), the N5 norm-budget law does not extend to multi-vector
stacking and a more complex predictor is required. If capping at the 75th
percentile does not prevent logPPL collapse (logPPL_cap >= 6.5 in any tested
configuration), the cap setting is wrong and needs recalibration. Either
failure falsifies this hypothesis.

---

## 4. Citations (Citation Rigor >= 80 words)

```
[N5 geometry result — this project, C2, SUPPORTED]: logPPL = 5.40 + 2.87 *
offshell_R, R² = 0.81 (23 rows, Gemma-2-2B) — the foundational law being
extended to multi-vector stacking; the parameters (slope 2.87, intercept 5.40)
are the direct inputs to the master-curve prediction.

[N17 geometry result — this project, C2, SUPPORTED]: Spearman(offshell,
logPPL) = +0.71 — the single-vector rank correlation; E22's multi-vector
master curve should achieve similar or higher rho (more data points, more
discriminative range of offshell values).

Korznikov, A., et al. 2026 ICML 'The Rogue Scalpel: Activation Steering
Compromises LLM Safety' (arXiv:2509.22067) — Guard Layer B (norm/manifold
clamp): "Cap the edit so the steered state stays in-distribution; if
||alpha*v_safe|| > beta*mu(l): rescale to beta*mu(l)." E22 is the multi-
vector empirical test of this guard; beta ~ 0.5-0.75 corresponds to the
q-quantile used here.

[Steering-stackable-vs-competing-analysis.md §3.3, this project]: "Steered
LLM Activations are Non-Surjective (2604.09839) proves steered states can lie
off the manifold reachable by any prompt — stacking too many additive edits
compounds this. IDS (2510.13285) exists specifically to keep stacked edits
in-distribution." — the IDS method and the norm-budget cap are complementary
mitigations for the same off-manifold stacking problem.
```

---

## 5. Mechanism

### 5.1 Master collapse curve prediction

From N5 (SUPPORTED):

    logPPL(h_steered) = 5.40 + 2.87 * (||h_steered - h_original|| / ||h_original||)

For a multi-vector stack:

    ||delta h|| = ||sum_i alpha_i v_i||

The master curve is:

    logPPL(N-vector stack) = 5.40 + 2.87 * ||sum_i alpha_i v_i|| / ||h||

This predicts that ALL multi-vector experiments (E17: 2 vectors, E18: 2-5 vectors,
E21: Conceptor AND, E22: capped) fall on the SAME N5 curve when plotted against
the total normalized displacement.

### 5.2 Norm cap implementation

```python
def cap_delta(delta_h, h, quantile=0.75):
    """Cap delta_h so ||delta_h|| <= q-quantile of ||h|| at injection layer."""
    h_norm = torch.norm(h, dim=-1, keepdim=True)  # per-token
    budget = h_norm * quantile                      # 75th percentile proxy
    delta_norm = torch.norm(delta_h, dim=-1, keepdim=True)
    scale = torch.clamp(budget / (delta_norm + 1e-8), max=1.0)
    return delta_h * scale
```

The budget B = quantile(||h||) is estimated once from the natural activation
distribution at the injection layer (pre-computed; no online cost).

### 5.3 Budget quantile calibration

From N5: logPPL collapse begins at offshell ~ 0.4 (where logPPL > 6.5).
Setting B = 0.3 * ||h|| (30th percentile) ensures offshell stays below 0.4.
For practical use, B = 0.5 * mean(||h||) is the Rogue Scalpel Guard B
recommendation (beta ~ 0.5–0.75).

---

## 6. Predicted Delta

| Metric | Predicted value | Rationale |
|---|---|---|
| Spearman rho (offshell/||h||, logPPL) across all E17-E22 | >= 0.75 | N5 law extension |
| N5 slope (multi-vector) | [2.5, 3.2] | Within 10% of N5's 2.87 |
| N5 intercept (multi-vector) | [5.0, 5.8] | Within 10% of N5's 5.40 |
| logPPL with cap (B=0.3*||h||) | < 6.5 | Collapse threshold |
| logPPL without cap (N=5 stack at high alpha) | > 7.0 | N5 prediction at large offshell |
| Behavior efficacy (capped vs uncapped) | [75%, 95%] | Cap reduces alpha proportionally |

The N5 (SUPPORTED) and N17 (SUPPORTED: Spearman 0.71) geometry results directly
ground these predictions. Pre-registered; [NEEDS VERIFICATION] multi-vector.

---

## 7. Experimental Protocol

### 7.1 Primary experiment

- **Model:** Gemma-2-2B-it, 4-bit; smoke on Gemma-3-1B-it
- **Data pool:** collect all (offshell, logPPL) pairs from E17, E18, E21 runs
  plus dedicated E22 sweeps of (N=2,3,4,5) at multiple alpha values
- **Cap conditions:** uncapped vs capped at B = {0.3, 0.5, 0.75} * ||h||
- **Metrics:** Spearman rho (offshell, logPPL); N5 curve fit (slope, intercept);
  logPPL collapse rate (fraction of conditions with logPPL > 6.5)
- **Master curve plot:** all E17–E22 data points on one (offshell, logPPL) figure
- **Seeds:** 3 per (N, alpha, cap) condition
- **Wall-clock:** ~4 h on 4090 (requires E17–E21 data as input)

### 7.2 Where it shines

The master curve has the most discriminative power when the offshell range is
wide (from near-0 to > 0.5). Including both near-orthogonal (low offshell) and
anti-aligned (high offshell) vector pairs in the data pool maximizes the dynamic
range. The cap experiment is most useful when uncapped conditions are shown to
have logPPL > 6.5 (demonstrable collapse).

---

## 8. Cross-References

- **N5** (norm budget, SUPPORTED): the foundational law; E22 is its extension
- **N17** (concentration penalty, SUPPORTED: Spearman 0.71): single-vector
  evidence for the offshell-logPPL correlation
- **N16** (radial/angular, SUPPORTED): the "radial" component of delta_h is the
  off-shell displacement; E22 measures exactly this radial component
- **E17** (2-vector stacking): data for the master curve at N=2
- **E18** (Gram mass): Gram mass predicts the offshell growth; E22 measures
  the consequence (logPPL)
- **E21** (Conceptors): Conceptor AND's lower offshell should place it lower
  on the master curve than summed vectors at matched behavior
- **E47** (gate + ortho-stack + norm-cap): E22's cap is the "norm-cap" component
- **Rogue Scalpel Guard B** (corpus/steering-first-principles-v2): the norm cap
  is exactly Guard Layer B; E22 provides the empirical calibration

---

## 9. Committee Q&A

**Q: Does the N5 master curve assumption require that ALL multi-vector conditions
fall on the SAME curve?**

> Yes — the master curve prediction is that the ONLY relevant predictor of
> logPPL is total offshell displacement, regardless of how it is generated.
> If Conceptor AND (which operates angularly) produces LOWER logPPL than
> additive summation at the SAME offshell, the master curve is not universal
> and the angular/radial distinction (N16) is the missing variable. In that
> case, the master curve should be stratified by operation type (additive vs
> rotational/projection).

**Q: Is the 75th percentile the right cap level?**

> The 75th percentile of ||h|| is a heuristic. The N5 law calibrates collapse
> at offshell ~ 0.4. The 75th percentile of ||delta h||/||h|| needs to be
> confirmed to fall below 0.4 on natural activations. The experiment sweeps
> {0.3, 0.5, 0.75} to identify the right cap empirically.

---

## 10. Verification Checklist

- [ ] E17, E18, E21 (offshell, logPPL) data points collected and pooled
- [ ] Dedicated E22 sweep: N in {2,3,4,5}, alpha grid, uncapped and capped
- [ ] Master curve plotted: all E17–E22 points on (offshell/||h||, logPPL)
- [ ] Spearman rho computed and compared to 0.75 threshold
- [ ] N5 slope and intercept from multi-vector fit reported (vs N5's 2.87/5.40)
- [ ] Cap at {0.3, 0.5, 0.75} * ||h|| applied; logPPL collapse rate reported
- [ ] Optimal cap level (lowest logPPL without excessive efficacy loss) identified
- [ ] IDEA_TABLE.md row E22 updated; result in EXPERIMENT_LEDGER.md

---

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: UNTESTED / PENDING. Geometry
  grounding is strongest of all Block C hypotheses: N5 (logPPL = 5.40 +
  2.87 * offshell, R² = 0.81, SUPPORTED), N17 (Spearman 0.71, SUPPORTED),
  N16 (radial governs additive logPPL, R² = 0.81, SUPPORTED) all directly
  predict the master curve and the cap's effectiveness. The norm-cap is
  also Guard Layer B of the Rogue Scalpel mitigation. E22 is the most
  geometry-grounded experiment in Block C. Blocked on multi-vector injection
  and E17/E18/E21 data being available to pool. Code needed: cap_delta
  function (trivial); master-curve pooling from E17–E21.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-C.*

### Prior plausibility

HIGH. The N5 law is SUPPORTED with R² = 0.81 and N17 with Spearman 0.71.
The multi-vector extension is the most natural prediction: if offshell = f(N,
Gram mass, alphas), and logPPL = N5(offshell), then the master curve follows
by composition. The main uncertainty is whether Conceptor AND (angular) falls
on the SAME curve as additive vectors — if not, the master curve must be
stratified by operation type.

### Mechanism scrutiny

The N5 law is established empirically. The multi-vector extension is a
well-motivated prediction. The cap implementation is mechanistically sound.

### Confounds

1. **Conceptor AND's offshell:** the formula for ||delta h|| under Conceptor
   AND is not alpha * ||v|| — it is ||(C_AND - I) * h||, which depends on h.
   Computing this correctly requires measuring the actual post-projection
   displacement, not the projected vector norm.
2. **Layer-dependence:** the N5 curve parameters (2.87, 5.40) were fit at a
   specific layer; the cap budget B = q-quantile(||h||) is layer-dependent.
   The experiment should report which layer is used for both the N5 fit and
   the cap calibration.

### Verdict

**NOVEL+TESTABLE** with the strongest geometry grounding in Block C.
The master-curve prediction is the key falsifiable claim; the cap experiment
is its practical application. The Conceptor-AND offshell computation is the
main technical care point.

---

## Pseudocode & Methodology

This section specializes [`../METHODOLOGY.md`](../METHODOLOGY.md) to the multi-vector
norm-budget master curve + the quantile cap (Rogue Scalpel Guard B). Status:
`UNTESTED` — the cap is trivial; the master curve pools E17/E18/E21 data which must
first exist.

### 1. Steering-vector recipe (any stack + a norm cap)

```python
# Vectors are the usual closed-form DiffMean stack (METHODOLOGY §1.3); the novelty is the cap.
v_stack = sum(alpha_i * normalize(v_i) for v_i in V)        # ||Δh|| = ||v_stack||

def cap_delta(delta_h, h, quantile=0.75):                   # §5.2 — Guard Layer B clamp
    budget     = norm(h, dim=-1, keepdim=True) * quantile   # B = q-quantile of ||h||
    delta_norm = norm(delta_h, dim=-1, keepdim=True)
    scale      = clamp(budget / (delta_norm + 1e-8), max=1.0)
    return delta_h * scale                                  # rescale so ||Δh|| <= B
```

### 2. Experiment procedure

```text
# Pool offshell/PPL from prior stacks + a dedicated E22 sweep:
data = collect_rows(E17, E18, E21)            # (offshell/||h||, logPPL) pairs
for N in {2,3,4,5}:
  for alpha in alpha_grid:
    for cap in {uncapped, 0.3, 0.5, 0.75} * ||h||:
      delta = v_stack;  if cap: delta = cap_delta(delta, h, cap)
      inject via hooks.apply_operation (add/relative_add)
      log: geometry.offshell_displacement, geometry.norm_budget,  # ||Δh||/||h||
           PPL (logPPL), behavior_efficacy
master_rho = spearman(offshell_over_h, logPPL)         # across ALL pooled rows
fit slope,intercept = linreg(offshell_over_h, logPPL)  # compare to N5: 2.87 / 5.40
```

### 3. Measurement & decision rule

- PRIMARY metric: Spearman ρ(offshell/‖h‖, logPPL) across all E17–E22 rows (the
  master collapse curve) AND the capped-logPPL collapse rate.
- Pre-registered FALSIFIER (§3): ρ `< 0.75` (N5 does not extend to multi-vector) OR
  capping at the 75th percentile fails to keep `logPPL < 6.5` in any tested config ⇒ FALSIFIED.
- Targets (§6): fitted slope `[2.5,3.2]`, intercept `[5.0,5.8]` (within 10% of N5).

### 4. Where the code is / status

`cap_delta` is a few lines and `geometry.offshell_displacement` / `norm_budget`
already exist. The blocker is **data availability**: the master curve needs E17/E18/E21
rows (themselves blocked on multi-vector injection). Caveat (Addendum confound 1):
the Conceptor-AND offshell must be measured as the actual post-projection
displacement `||(C_AND−I)h||`, not `alpha*||v||`. `UNTESTED`.

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

Full per-hypothesis provenance (exact experiments, reproduce commands, artifact links, reasoning trace): [`PROVENANCE/E22.md`](../PROVENANCE/E22.md).

- **Experiments:** analysis campaign (computed quantities in the campaign JSON; see the provenance file).
- **Reproduce:**

```bash
PYTHONPATH=src python scripts/campaign_sweep.py --model models/google/gemma-3-270m-it --quant none --hyp E22 --tag-prefix E22-budget --layers 16 --alphas 0.02 0.05 0.1 0.2 0.4 --ops relative_add --behaviors anger happiness  # cumulative ||sum alpha_i v_i|| swept against PPL to find the collapse knee
```
