# Pre-registration — stacking hill-climb (lesson 12)

**Written BEFORE any measurement in this run.** Governing rule: CLAUDE.md §9
(stacking discipline). Screening tier — small n, single seed, judge-scored
(off-family Qwen2.5-3B, ROC-AUC 0.665–0.751 → below the 0.85 bar; see
`../JUDGE_VALIDITY.md`). Nothing here is an evaluation-tier claim.

## 0. What was already known before this pre-registration

`artifacts/results.json` (the prior run, n=150/rung) is on disk and I have read
it. So the rungs `[A]`, `[A,B]`, `[A,B']`, `[A,B,B']` are **replications**, not
blind predictions, and are labelled as such below. The genuinely un-run cells
are: prior **C**, prior **CLAMP**, prior **GATE**, every **standalone** effect
except B's, the **unsteered baseline**, and the whole **benign arm**.

## 1. Site inventory — every prior this lesson can actually run

Each prior is named by its intervention SITE (tower / layer / operation /
training signal). Only priors runnable from this lesson's existing code plus a
small addition (`hillclimb.py`) are listed.

| id | site (tower · layer) | operation | training signal | source |
|---|---|---|---|---|
| **A** | residual stream, output of decoder block **12** | `relative_add`: `h ← h + α‖h‖v̂`, α=0.08 | CAA diff-of-means over 300 harmful vs 300 benign last-token acts @L12 (arXiv:2312.06681) | existing `build_priors` |
| **B** | residual stream, output of decoder block **8** | `relative_add`, α=0.08 | *same vector as A* (direction held fixed so only SITE varies) | existing `build_priors` |
| **B′** | residual stream, output of block **12** — the SAME site as A | `add` (literal ActAdd, raw `h ← h + αv`), α rescaled so B′ alone ≈ A alone in magnitude | *same vector as A* | existing `build_priors` |
| **C** | residual stream, output of block **12** — SAME site as A, **orthogonal direction** | `relative_add`, α=0.08 | top PC of the pooled extract activations @L12 **after projecting out the refusal direction** (Gram–Schmidt ⇒ cos(C, v_refusal)=0 by construction) | NEW in `hillclimb.py` |
| **CLAMP** | same layers as the live injectors, but runs **after** every injector hook in the same forward pass | constraint, not injection: `δ ← δ·min(1, cap/r)` where `r = ‖δ‖/‖h_base‖`, cap=0.10 per layer | none — a geometric guard (Rogue-Scalpel guard layer B / the N5 norm budget) | NEW in `hillclimb.py` |
| **GATE** | **before** the forward pass; reads mean-pooled L12 activations and multiplies the whole stack by 0/1 | condition (CAST) | lesson-1 MLP probe, trained on harmful vs benign L12 mean-pooled acts | lesson 2 `gate.HarmGate` |

Deliberately **excluded** (cannot be run honestly from this lesson): rotational /
angular steering (no rotation op exists in `model_utils`), ReFT-style learned
interventions (different lesson, needs training), a second *labelled concept*
vector (this lesson's pool has only the harmful/benign contrast — C is an
orthogonality control, not a second concept, and is labelled as such).

## 2. Pre-registered STACK / COMPETE matrix (§9 rule, applied before measuring)

Rule: *different site ⇒ STACK; same site + same direction + different operation
⇒ COMPETE; near-orthogonal directions ⇒ STACK until the norm budget is spent.*

|  | A | B | B′ | C | CLAMP | GATE |
|---|---|---|---|---|---|---|
| **A** | — | **STACK** (different site L12/L8) | **COMPETE** (same site, same direction, different op) | **STACK\*** (same site, orthogonal direction — until budget spent) | **STACK** (meta: constraint, not an injector) | **STACK** (meta: disjoint stage, pre-forward) |
| **B** | | — | **STACK** (different sites L8/L12) | **STACK** (different sites) | **STACK** (meta) | **STACK** (meta) |
| **B′** | | | — | **STACK\*** (same site, orthogonal directions; B′ is not norm-aware so budget-limited) | **STACK** (meta — the clamp is precisely the fix for B′'s norm blow-up) | **STACK** (meta) |
| **C** | | | | — | **STACK** (meta) | **STACK** (meta) |
| **CLAMP** | | | | | — | **STACK** (two meta-layers, disjoint stages: gate decides IF, clamp bounds HOW MUCH) |
| **GATE** | | | | | | — |

`STACK*` = STACK by the direction clause but explicitly **conditional on the
norm budget**; the §9 rule says these stack *until the budget is spent*.

**Exactly one unconditional COMPETE is predicted: A × B′.**

## 3. The ladder actually built (each rung adds exactly ONE new prior)

The all-on hybrid is **forbidden** (§9), so the ladder contains only priors that
were pre-classified STACK. The single pre-classified COMPETE pair is measured as
a **separate control**, never folded into the ladder.

```
 R0  []                                  unsteered reference
 R1  [A]                                 base prior
 R2  [A, B]                     + B      new prior: DISJOINT SITE       (site clause)
 R3  [A, B, C]                  + C      new prior: ORTHOGONAL DIRECTION (direction clause)
 R4  [A, B, C] + CLAMP          + CLAMP  new prior: NORM-BUDGET CONSTRAINT (meta)
 R5  [A, B, C] + CLAMP + GATE   + GATE   new prior: CAST CONDITION        (meta)
```

Controls (measured, outside the ladder): `[B]`, `[C]`, `[B′]` standalone, and
`[A, B′]` — the pre-registered COMPETE pair.

**Note on the existing lesson:** its `rung3 = [A, B, B′]` *is* the all-on hybrid
and it mixes a pre-classified COMPETE pair. It is built deliberately as a
demonstration of the forbidden configuration; this hill-climb does not extend it.

## 4. Competition test (how a contradiction is detected)

For a prior P added at rung k:

```
 marginal(P)   = metric(rung k) − metric(rung k−1)
 standalone(P) = metric([P] alone) − metric(R0, unsteered)
 COMPETITION if  marginal(P) < standalone(P)   (the prior delivers less inside
                                                the stack than it does alone)
```

## 5. Numeric predictions (registered before the run)

| # | prediction | falsifier |
|---|---|---|
| **P1** (replication) | R2 marginal refusal ≤ 0 and gibberish rises vs R1 — the site clause is **overridden** by the norm-budget clause because A alone is already at gibberish ≈ 0.43, i.e. there is no coherence headroom to add into | R2 marginal refusal > +0.05 |
| **P2** (novel) | C standalone raises gibberish by < 0.15 over R0 and moves refusal by < 0.10 — an orthogonal direction is behaviourally near-inert but still spends budget | C standalone refusal shift > 0.15 |
| **P3** (novel) | R3 (adding orthogonal C) costs **less** coherence per unit budget than R2 did (the direction clause is weaker than the site clause here): gibberish(R3) − gibberish(R2) < gibberish(R2) − gibberish(R1) | the reverse ordering |
| **P4** (novel) | CLAMP **stacks**: gibberish(R4) < gibberish(R3) by ≥ 0.10 while \|refusal(R4) − refusal(R3)\| < 0.10 | clamp cuts refusal by > 0.10 ⇒ CLAMP **competes**, contradicting the matrix |
| **P5** (novel) | GATE **stacks**: on harmful, \|R5 − R4\| < 0.05 on refusal (probe fires on nearly all harmful prompts); on **benign**, R5 gibberish drops toward R0 by ≥ 0.15 | harmful refusal moves > 0.10 ⇒ GATE is not a clean meta-layer here |
| **P6** (replication) | `[A,B′]` shows no gain over the **best single** prior — and the best single is to be read against **all** standalones, not only A | — |

## 6. Known-issue check registered here too

`artifacts/results.json` carries `single.B_refusal_rate = 0.2533` (B alone,
L8) which is **higher** than the ladder's own base rung `[A]` at 0.20, and is
absent from the README. Prediction: the re-measured standalone ordering will
again put **B above A**, meaning the ladder is anchored on the *worse* of the two
single priors and every "marginal" in the README is measured from the wrong base.

---
*Screening tier. Single seed. Judge below its own validity bar. No claim here is
evaluation-tier or external-ready.*
