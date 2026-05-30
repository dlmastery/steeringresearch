---
name: steering-combo-ladder
description: >
  Use when designing, running, or reviewing multi-vector stacking experiments
  for activation steering. Covers the stack-vs-compete decision rule, the
  norm-budget cap (N5), CAST conditioning as a meta-layer, the additive 2→N
  ladder protocol, and the forbidden all-on hybrid. Renders the pairwise
  decision matrix. Steering instantiation of autoresearch-combo-ladder.
---

# Skill — steering-combo-ladder

## When to use

Use this skill whenever you are:

- Deciding whether two steering methods stack or compete.
- Designing a new combo experiment (which priors to combine, in which order).
- Reviewing a stacking run for the forbidden all-on hybrid.
- Checking the norm-budget usage before adding a new vector to a stack.
- Verifying that CAST conditioning is treated as a meta-layer (not a peer
  method) in a stacking design.
- Adding a row to the additive 2→N ladder and measuring the marginal effect.
- Reporting the stack/compete matrix on the master dashboard.

This skill depends on:
- `../../skills/steering-intervention-lib/SKILL.md` (norm cap, hook
  composition, SteeringContext nesting)
- `../../skills/steering-eval-bundle/SKILL.md` (JailbreakBench CR on every
  combo row)
- `../../skills/steering-rogue-scalpel-guard/SKILL.md` (guard layers A–E
  must wrap every new vector added to the stack)

This is the steering domain's instantiation of the abstract combo ladder in
`../../meta-skills/autoresearch-combo-ladder/SKILL.md`. That skill owns the
abstract protocol ("stack only orthogonal axes, additive ladder, all-on
forbidden"); this skill owns the steering-specific decision rules, the
decision matrix, and the norm-budget arithmetic.

---

## 0. The core organizing principle

Stackability is governed by WHERE in the forward pass and HOW methods act
there. This is a LINEAR-ALGEBRA fact, not a heuristic:

```
Two vectors v1, v2 in the residual stream:
  v2 = (v2·v1/|v1|²) v1   +   v2_perp
        \_____________/         \______/
          INTERFERENCE          INDEPENDENT BEHAVIOR
```

The interference term competes for the same activation axis. The orthogonal
term is free to stack. The Gram matrix of the active steering vectors
(off-diagonal cosines) directly measures the interference budget.

```
Decision rule (derived, not memorized):
  different site                  → STACK  (disjoint writes)
  same site + near-orthogonal dir → STACK  (until norm budget spent)
  same site + same dir + diff op  → COMPETE (add vs rotate on same plane)
  same site + aligned dir + same op → COMPETE (double-count; norm waste)
  any combination that blows N5  → COMPETE (budget exhausted)
```

---

## 1. The seven-axis classification of stack vs compete

From `corpus/steering-first-principles-v2-with-PSR-and-rogue-scalpel.md`
and `corpus/steering-missed-dimensions-and-highdim-algebra.md`:

Two methods STACK iff they differ on Axis 1 (WHERE — intervention site) OR
are near-orthogonal on Axis 2 (WHAT — direction).

Two methods COMPETE iff they are on the same Axis 1 site AND the same Axis 2
direction AND differ on Axis 4 (HOW — operation), or if they jointly exhaust
the norm budget (N5, Axis 3).

Axis 5 (WHEN — conditioning / CAST) is the META-axis: it wraps any
combination of Axes 1–4 without consuming budget. See §4.

---

## 2. The pairwise decision matrix

Legend: STACK = compose cleanly; CARE = stack with orthogonalization or norm
monitoring; COMPETE = pick one, do not run both.

| | CAA / ActAdd (additive, residual) | Angular / Selective (rotational, residual) | SAE-feature steer (additive, residual) | KV-cache steer | Attention-score (PASTA / SpotLight) | DoLa (decoding) | CAST gate (meta-layer) |
|---|---|---|---|---|---|---|---|
| **CAA / ActAdd** | CARE (orthogonalize; monitor norm budget N5) | COMPETE (same residual plane, incompatible op) | CARE (use SAE-TS to de-conflict; monitor N5) | STACK (disjoint site) | STACK (disjoint site) | STACK (post-stack at decode) | STACK (meta-layer; no budget consumed) |
| **Angular / Selective** | COMPETE | CARE (multi-plane only; no shared plane) | COMPETE (same residual subspace, diff op) | STACK | STACK | STACK | STACK |
| **SAE-feature steer** | CARE | COMPETE | CARE (multi-feature; N5 shared) | STACK | STACK | STACK | STACK |
| **KV-cache steer** | STACK | STACK | STACK | CARE (SKOP to avoid contamination) | CARE (shared attn pathway) | STACK | STACK |
| **Attention-score** | STACK | STACK | STACK | CARE | COMPETE (PASTA vs SpotLight on same heads) | STACK | STACK |
| **DoLa (decoding)** | STACK | STACK | STACK | STACK | STACK | — | STACK |
| **CAST gate** | STACK | STACK | STACK | STACK | STACK | STACK | CARE (compose conditions via AND/OR; not multiplicative gate) |

**How to read CARE cells:** CARE means "stack is possible but requires
explicit management of one of: (a) near-orthogonality of the direction
vectors (check Gram matrix off-diagonal cosines); (b) the cumulative norm
budget N5; (c) attention/KV pathway contamination via SKOP."

**Source:** synthesized from `corpus/steering-stackable-vs-competing-analysis.md`
§3 and §4, plus first-principles reasoning from §0 of this skill.

---

## 3. The norm-budget cap (N5)

The norm budget is the primary constraint on how many vectors can be stacked
additively on the residual stream before the cumulative displacement pushes
h off the data manifold (hypothesis N5 from `corpus/steering-missed-
dimensions-and-highdim-algebra.md`).

**Definition:**
```python
# For k stacked vectors at the same layer:
cumulative_budget = sum(|alpha_i * v_i_safe|, i=1..k) / mu_l
# where mu_l = mean activation norm at layer l (pre-cached)
# and v_i_safe = the sanitized (guard-A-projected) version of v_i
```

**Hard cap:** cumulative_budget < 1.0. The CLAMP operation in
`src/steering/hooks.py` enforces this for each individual vector (guard layer
B); but for STACKED vectors, the cumulative budget must be verified by the
combo ladder code before adding a new vector.

**Budget allocation protocol:**
1. Compute the budget consumed by the current stack (all active vectors).
2. Estimate the budget that the candidate new vector would add.
3. If adding the candidate would exceed 0.85 of the budget (leave 15% margin),
   either reduce its alpha or do not add it.
4. Log the per-step budget usage in the experiment ledger.

**The interference budget (hypothesis N18):**
Beyond the norm cap, stacking k near-orthogonal vectors accumulates second-
order interference proportional to the sum of |cos(v_i, v_j)| overlaps:
```python
gram_matrix = v_stack @ v_stack.T         # (k, k) cosine matrix
off_diagonal_mass = gram_matrix - eye(k)  # interference budget
interference = |off_diagonal_mass|.sum()  # should be << k for clean stacking
```
Interference below 0.1 * k means the vectors are effectively orthogonal.
Interference above 0.3 * k means the stack is degraded — consider
orthogonalization via SAE-TS before proceeding.

---

## 4. CAST conditioning as a meta-layer

**The key insight (corpus/steering-first-principles-v2 §4):** CAST-style
conditioning is NOT a peer steering method — it is a META-LAYER. The gate is
a READ-only similarity probe at an early layer; the injection is the standard
additive edit at a later layer. These are different axis-1 sites (read vs
write), different operations (probe vs modify), consuming zero norm budget.

**Consequence:** CAST stacks on top of essentially every residual-stream
behavior method. It does not consume the norm budget; it does not interfere
with any direction in the injection subspace; it can be AND/OR composed with
other condition probes.

**In the combo ladder:** CAST is added as a "free" meta-layer that wraps the
entire additive stack. It appears in every combo-ladder row because it imposes
no additional cost and improves selectivity (Axis 5). Its only resource
consumption is one cosine probe per early-layer forward pass.

**Composition of multiple CAST conditions:**
```python
# OR composition: steer if ANY condition fires
gate = any(cos(h_early, v_cond_i) > theta_i for i in range(n_conditions))
# AND composition: steer only if ALL conditions fire
gate = all(cos(h_early, v_cond_i) > theta_i for i in range(n_conditions))
```

When composing multiple conditions via OR/AND, validate on the CAST condition
sets (OR/AND multi-condition mixtures) from the STANDARD eval tier.

---

## 5. The additive 2→N ladder

**Protocol:** Each row of the combo ladder adds exactly ONE new orthogonal
prior on top of the previous row. This makes the marginal effect of each
prior readable and attributable.

```
Row 0:  baseline (no steering; unsteered Gemma-2-2B)
Row 1:  + Behavior vector A (tuned at single-prior level, Rung-3 cleared)
Row 2:  Row 1 + Behavior vector B (orthogonal to A; different axis or site)
Row 3:  Row 2 + CAST conditioning gate (meta-layer; wraps Row 2)
Row 4:  Row 3 + Guard A+B+C+D+E (safety guard wrapping everything)
...
Row N:  Row N-1 + Behavior vector N (must pass orthogonality check vs all prior vectors)
```

**Requirements for adding a new vector to the stack:**

1. The new vector must have been hill-climbed at the SINGLE-PRIOR level
   (Rung-3 cleared on its own, per the tiered ladder in
   `../../skills/steering-tiered-ladder/SKILL.md`) before entering the combo.
   Do not stack untuned vectors.

2. The new vector must satisfy the orthogonality check:
   ```python
   for existing_v in current_stack:
       assert abs(cosine_similarity(new_v, existing_v)) < 0.3
   # Threshold 0.3 is the default; stricter (0.15) for safety-critical combos
   ```
   If the check fails, orthogonalize the new vector via SAE-TS before adding.

3. Adding the new vector must not exceed the norm budget (§3).

4. JailbreakBench CR is re-measured on the new stack row. CR > 0% = DO NOT
   add this vector; demote the combo row.

5. The marginal effect of the new vector (delta behavior score, delta
   capability, delta coherence) is computed as (Row N metric − Row N-1
   metric) for each axis. Report the marginal, not just the cumulative.

---

## 6. The forbidden all-on hybrid

**The all-on hybrid is forbidden at every rung, at every stage of the
project.** Turning every available vector on simultaneously produces:

1. An unreadable result: the marginal effect of each prior is confounded with
   every other prior.
2. An almost-certain norm-budget violation: the cumulative displacement
   pushes h off the manifold; coherence collapses and CR likely increases.
3. A violation of the attribution discipline: if the composite improves (or
   regresses), you cannot credit (or blame) any individual method.

This is the steering analogue of the `sg_full_fib` result in the meta-
process corpus (CLAUDE.md §9): the "everything on" hybrid was observed to
underperform curated additive stacks by ~11.5 pp in the meta-process
literature.

**If you are tempted to run the all-on hybrid:**
- You are treating the combo as a search, not a scientific question.
- You have not hill-climbed the individual priors.
- You are confused about which axis the gain is coming from.

Fix: hill-climb each prior individually → confirm orthogonality → add one
at a time → measure marginal effects → stop when budget is spent.

---

## 7. Recommended stack for multiple non-interacting safety vectors

Concrete recipe for the Gemma-2-2B on a single 4090 (based on §5 of
`corpus/steering-stackable-vs-competing-analysis.md`):

1. **Meta-layer (CAST, free):** one condition probe per safety property
   (hate-speech, self-harm, illegal-advice), composed with OR. Read-only
   early-layer probes. Zero budget consumed.

2. **Behavior injection (STACK):** one additive CAA-style vector per
   safety property, orthogonalized via SAE-TS (so inter-vector |cos| < 0.15).
   Each vector individually passes Rung-3; combined norm budget < 0.7.

3. **Coherence guard (STACK, implicit):** Guard Layer B (norm clamp at each
   injection site) is always active. IDS-style in-distribution projection is
   the optional upgrade for Rung-4.

4. **Safety guard (STACK, wraps everything):** Guard layers A+B+C+D+E from
   `../../skills/steering-rogue-scalpel-guard/SKILL.md` wrap the full stack.

5. **DO NOT** also rotate the same planes (Angular/Selective steering) — the
   decision matrix says COMPETE. Choose additive OR rotational.

6. **Optional free upgrade:** DoLa contrastive decoding at decode time (after
   all residual edits, disjoint site; STACK by construction).

**What competes with this stack:** plain prompting often matches or beats
naïve SAE steering (AxBench result); use SAE vectors only when curated and
orthogonalized. Avoid stacking so many vectors that MMLU regresses > 2 pp
(O'Brien et al., arXiv:2411.11296, report this over-refusal/over-capability-
loss failure mode [NEEDS VERIFICATION on Gemma]).

---

## 8. Running a combo-ladder experiment (step-by-step)

1. **Pre-check:** verify all single-prior vectors are Rung-3 cleared and
   hill-climbed. If any are not, stop and hill-climb them first.

2. **Orthogonality matrix:** compute the Gram matrix of all candidate vectors.
   Flag any pair with |cos| > 0.3 as a COMPETE pair; resolve via SAE-TS
   orthogonalization or choose one over the other.

3. **Budget arithmetic:** compute the sum of all alpha_i * |v_i_safe| norms.
   Confirm cumulative budget < 0.85.

4. **Construct the additive ladder:** build each row by adding one vector.
   For each row:
   a. Apply the stack with SteeringContext (nested or composed).
   b. Run the SMOKE eval tier (AxBench-mini + MMLU-500 + WikiText-ppl +
      JailbreakBench).
   c. If CR > 0%: stop; do not proceed to the next row; log COMPETE.
   d. If any axis regresses: diagnose (which vector caused it) and stop.
   e. Log the marginal effect per axis.

5. **CAST gate:** add the CAST meta-layer as a separate row AFTER the first
   behavior vector is confirmed. Measure the selectivity improvement.

6. **Guard layers:** add A+B+C+D+E as the final row (or confirm they were
   active throughout — recommended). Verify the universal attack (V3) is
   neutralized at this row.

7. **Render the decision matrix** on the master dashboard and the per-
   hypothesis sub-dashboard; every combo row must appear as a separate
   experiment with its own ledger entry.

---

## 9. Dashboard rendering of the combo ladder

The master dashboard must include the **stack/compete matrix** live-rendered
from the experiment data (CLAUDE.md §11). It must show:

- The pairwise decision matrix (§2 of this skill) as a sortable table.
- The additive ladder rows with marginal effects per axis.
- The norm-budget gauge per row (bar chart: consumed / cap).
- Gram-matrix off-diagonal cosines for the active stack (heatmap).

All cells must carry `n=X` + SCREENING/EVALUATION tier chips. No bare numbers.

---

## Hard rules

1. NEVER run the all-on hybrid at any rung.
2. NEVER add a vector to the stack that has not first been hill-climbed at
   the single-prior level (Rung-3 cleared individually).
3. ALWAYS re-measure JailbreakBench CR after adding each new row to the stack.
   CR > 0% on any row = DISCARD that row; do not continue the ladder.
4. ALWAYS verify |cos(v_new, v_existing)| < 0.3 for all existing vectors
   before adding a new vector. Orthogonalize if needed.
5. ALWAYS check the cumulative norm budget before adding a new vector.
   Budget > 0.85 = STOP; reduce alpha or drop the vector.
6. CAST conditioning is a META-LAYER: it does not count as a "peer method"
   in the additive stack and is added as its own meta-row.
7. Stack ONLY vectors whose single-prior versions were tested on the SAME
   model (Gemma-2-2B) at the SAME rung. Do not combine a Rung-3 vector with
   a Rung-1 vector.

---

## Anti-patterns

| Anti-pattern | Consequence | Do instead |
|---|---|---|
| Running the all-on hybrid | Unreadable result; norm-budget violation; attribution lost | Additive 2→N ladder, one vector per row |
| Stacking an untuned vector | Combo inherits the single-prior's suboptimality; marginal effect is unreliable | Hill-climb at single-prior level first |
| Adding CAA and Angular/Selective to the same residual plane | They compete (COMPETE cell in the matrix); double-counting + off-manifold | Choose additive OR rotational for each behavior plane |
| Forgetting to re-measure CR on each combo row | Safety leak from the stack goes undetected | CR is mandatory per row |
| Treating CAST as a peer method with its own budget cost | CAST consumes no norm budget; treating it as peer inflates the budget estimate | CAST = meta-layer; add it as a meta-row with zero budget cost |
| Stacking vectors with |cos| = 0.5 without orthogonalization | Interference degrades both behaviors; marginal effects are negative | Check Gram matrix; use SAE-TS to de-conflict |
| Measuring only the final row's composite | Can't read marginal effects; can't diagnose which vector caused a regression | Report marginal deltas per row, per axis |

---

## Cross-references

- Meta-process combo ladder (abstract protocol this skill instantiates):
  `../../meta-skills/autoresearch-combo-ladder/SKILL.md`
  (Note: pre-declared cross-reference path for the meta-pack author.)
- Hook composition and norm-budget clamp implementation:
  `../../skills/steering-intervention-lib/SKILL.md` §2.4, §4
- JailbreakBench CR protocol (mandatory per combo row):
  `../../skills/steering-eval-bundle/SKILL.md` §4
- Guard layers A–E (must wrap every combo stack):
  `../../skills/steering-rogue-scalpel-guard/SKILL.md`
- Single-prior hill-climb before stacking:
  `../../skills/steering-tiered-ladder/SKILL.md` §3
- Stack vs compete first principles:
  `corpus/steering-first-principles-v2-with-PSR-and-rogue-scalpel.md` §3 and §4
- Pairwise decision matrix source:
  `corpus/steering-stackable-vs-competing-analysis.md` §4
- Norm budget N5 and interference budget N18:
  `corpus/steering-missed-dimensions-and-highdim-algebra.md` §F-B, §F-F
- Source modules: `src/steering/hooks.py`, `src/steering/geometry.py`,
  `src/steering/eval.py`, `src/steering/runner.py`
