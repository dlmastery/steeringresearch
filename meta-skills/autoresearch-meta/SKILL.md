---
name: autoresearch-meta
description: Use any time you start or run a rigorous, publication-grade autoresearch program on a new topic — the portable spine that defines the Karpathy keep/discard core, the 7-step ritual, the 5-rung benchmark ladder, the Goodhart-resistant composite, the screening→hill-climb→evaluation funnel, the orthogonal-axis combo ladder, the hierarchically-linked dashboard, the dual-track audit, and the 10-step instantiation checklist.
---

# Skill — The autoresearch meta-process (the spine)

## When to use

Use this skill whenever you are setting up, running, or auditing a principled
autoresearch program on **any** topic that has these properties:

- the object of study has **more than one objective axis** that can trade off
  against each other (so a single scalar metric would be gameable);
- experiments are cheap enough to iterate many times, but expensive enough that
  you should not run the full benchmark to catch a plumbing bug;
- claims will eventually face third-party scrutiny, so every number must be
  pre-registered, statistically defended, and reproducible.

This is the spine. It names *what* to do and *why*; each supporting skill in
this pack owns the *how* of one workflow and is cross-referenced below. Read this
file cover-to-cover before instantiating the process on a new topic, then keep it
open as the index while you work.

Throughout, a **neutral worked mini-example** runs alongside the abstract rules:
a program studying **inductive biases in image-classification models** (e.g.,
"does a rotation-equivariance prior help small-data generalization without
hurting calibration?"). Everything in the example is illustrative; substitute
your own axes, datasets, and priors. A second neutral example —
**tabular-ML feature priors** — appears where a contrasting domain clarifies a
rule.

---

## 0. North Star

The deliverable of an autoresearch program is **not** the best artifact (the best
model, the best config). It is a **defensible body of evidence**: a sortable
master dashboard that drills into per-hypothesis and per-experiment
sub-dashboards, a findings ledger gated by statistical rigor, and an auditable
write-up. Transparency is the product; the winning weights are secondary.

The **methodology itself is a deliverable**. This pack is the portable encoding
of the process. The topic you are working on now is one *instantiation* of it.
When you improve the process here, the improvement ports to the next topic for
free.

### Goal / outcome clarity mandate (binding across all surfaces)

Every external-facing surface — README, master dashboard, paper/report — must
state the **concrete outcome** up front, not merely "this is a research program."
The concrete outcome means:

1. **What is being invented or discovered**: the specific trade-off being
   maximized (e.g., "maximize primary_efficacy while keeping cost_axis_1 and
   cost_axis_2 within acceptable bounds").
2. **What SOTA means concretely here**: the current best-known trade-off for this
   model class / task domain, and why it is the benchmark to beat.
3. **What a "winner" is**: the exact definition in terms of the composite metric
   and the Pareto condition (not a vague "best method").

**Anti-pattern:** "This project studies [domain] methods with the goal of
understanding which techniques work best." — This tells the reader nothing about
what to optimize or what winning looks like.

**Required pattern:** "This project discovers the [domain] configuration that
maximizes [primary_efficacy] while keeping [cost_axis_1] below [threshold] and
[cost_axis_2] below [threshold] — the best simultaneous trade-off known for
[model class / scope]. A 'winner' is any method that Pareto-dominates the current
champion on the composite metric (formula fingerprinted in every experiment)."

Apply this mandate at instantiation step 1 (before writing the axis taxonomy)
and verify it is present on every external-facing surface at the Checkpoint step
of every campaign milestone.

---

## 1. The Karpathy core invariant (and the three adaptations)

The origin is Karpathy's autoresearch loop: *modify → run (minutes) → check if
improved → keep/discard → repeat*, an agent running autonomously until
interrupted, one thing changed, one metric to beat.

**The core invariant we keep, verbatim:**

> **Always start from the current best config (the champion). Change exactly ONE
> thing. Keep it if and only if the composite improves at matched coherence.
> Revert otherwise. Never wander.**

The champion config is **sacred**. Every experiment is a single-axis perturbation
of it (or of a documented baseline when you are first probing a new axis). This
turns a sprawling search into a sequence of readable, attributable steps.

**Three adaptations distinguish this from vanilla Karpathy autoresearch:**

1. **Never deviate far from the winner.** Vanilla agents can try wild changes
   because a run is cheap. We do not: a wild change confounds the attribution of
   any gain. A documented **axis taxonomy** (the orthogonal axes of your topic —
   §9) tells you which axes are independent and therefore safe to perturb one at
   a time.
2. **The agent IS the expert researcher.** No blind search. Every experiment is
   Diagnose → Cite → Hypothesize → Predict → Execute → Analyse → Checkpoint (§2).
   Reasoning quality gates experiment quality. An Optuna sweep explores a space;
   this process *eliminates* most of the space with mechanistic reasoning and
   reads the primary literature for the *diagnosed* failure, not in general.
3. **Ladder-bound, not time-bound.** A method must clear the promotion gate at
   rung *k* before it may consume rung *k+1* compute (§3). You never run the
   expensive benchmark to find a bug the cheap one would have caught.

> *Worked example.* Champion = a baseline CNN at a fixed recipe. One experiment
> adds a rotation-equivariant first block (ONE axis: the filter-init/equivariance
> axis), holding optimizer, schedule, augmentation, and seed fixed. If the
> composite improves at matched calibration, KEEP; else revert and record why the
> mechanism prediction was wrong. You do **not** also change the learning rate in
> the same experiment — that is two axes and the attribution is lost.

---

## 2. The 7-step ritual (no experiment without it)

Each experiment authors a **pre-run** reasoning entry (steps 1–4) BEFORE launch,
then a **post-run** entry (steps 6–7) after. The runner re-validates the entry on
launch and **refuses to fill pre-run fields with placeholders** — a missing
diagnosis/citation/hypothesis/prediction is a protocol violation, not a TODO.

| # | Step | Gate | Minimum |
|---|---|---|---|
| 1 | **Diagnose** | word-count | ≥60 words; read the last `experiment_log` row; name the specific failure mode / open question; reference ≥1 prior experiment by tag or per-row metric |
| 2 | **Cite** | citation-format | ≥40 words single-paper / ≥80 multi; the exact paper motivating the change |
| 3 | **Hypothesize** | word-count + keyword | ≥50 words; must contain "mechanism" / "because" / "per [paper]" |
| 4 | **Predict** | word-count + numeric | ≥25 words; a numeric range on the composite + ≥1 sub-metric, stored BEFORE the run |
| 5 | **Execute** | one-change | exactly ONE axis changed from the champion; runner re-validates the entry |
| 6 | **Analyse** | word-count + verdict | ≥30 words; actual vs predicted; verdict `KEEP`/`DISCARD`/`NEAR-MISS`; composite to 4 dp; Δ vs global best; per-axis narrative |
| 7 | **Checkpoint** | word-count + artifacts | ≥40 words; update every dashboard/ledger artifact; commit + push |

**Citation format (mandatory, every citation):**

```
Author1, Author2, ..., YEAR VENUE 'Title'
(arXiv:XXXX.XXXXX) -- one-sentence relevance note.
```

Every identifier must be real. Mark `[UNVERIFIED]` if you are not certain the ID
resolves; mark inherited numbers `[NEEDS VERIFICATION]` until reproduced in your
own harness. A citation of the form `(Author2024)` with no title/venue/identifier
is refused by the gate.

**Hard rules of the ritual:** no `--bypass` flag (if a gate refuses, fix the
field); one config change per experiment; the experiment log is **append-only**;
the composite formula is SHA-256 fingerprinted and editing it breaks the project.

Owned by: [`../autoresearch-experiment/SKILL.md`](../autoresearch-experiment/SKILL.md)
and the reasoning-entry gates it inherits.

---

## 3. The benchmark ladder (CIFAR-style, 5 rungs)

> **Never run an expensive benchmark to find a bug a cheap one would catch.**

The ladder is the promotion mechanism. The **same axes are measured at every
rung — only the size and realism of the data/eval grow.** This is the property
that makes the ladder honest: a rung-1 SMOKE pass and a rung-4 FULL pass measure
the *same* objective vector, so a method's trajectory up the ladder is directly
comparable.

| Rung | Nickname | Cost/run | Proves | Gate to next rung |
|---|---|---|---|---|
| 0 | **UNIT** | seconds | plumbing works | the intervention changes the output; state restores exactly; tests green |
| 1 | **SMOKE** | 1–3 min | right direction | monotone effect in the intended direction + bounded coherence + no safety/constraint leak |
| 2 | **DEV** | 10–20 min | generalizes a little | beats baseline on held-out items at matched coherence |
| 3 | **STANDARD** | 1–3 h | real result | **Pareto-dominates** the prior method — no axis regresses past its gate |
| 4 | **FULL** | half-day+ | publication | full multi-axis win + ablations + red-team / negative-control neutralized |

**Promotion rule:** clear rung *k*'s gate before spending rung *k+1* compute. A
regression at any rung **demotes** the method with a logged `failure_reason`. The
ladder board on the dashboard shows, per method, the highest rung reached and the
gate it cleared or failed.

> *Worked example.* UNIT: a forward pass with the equivariant block runs and the
> shape/contract tests pass. SMOKE: on a 2-class, 500-image subset the prior
> improves accuracy without wrecking calibration. DEV: on the held-out classes it
> still beats the baseline CNN at matched ECE. STANDARD: on the full small-data
> split it Pareto-dominates the baseline across all axes. FULL: full dataset +
> ablating the equivariance away + a label-shuffle negative control that must
> fail. A method that fails calibration at SMOKE never consumes STANDARD compute.

Owned by: [`../autoresearch-tiered-ladder/SKILL.md`](../autoresearch-tiered-ladder/SKILL.md).

---

## 4. The composite metric (Goodhart-resistant, fingerprinted)

A multi-objective topic has **no single scalar**. The composite **must price
every objective axis** so a method cannot "win" by sacrificing one. The shape is
always: a primary efficacy term, minus a one-sided penalty for each cost axis
that should not regress.

```
composite = primary_efficacy
          - lambda_1 * max(0, regression_on_axis_1)   # e.g. capability / accuracy tax
          - lambda_2 * max(0, regression_on_axis_2)   # e.g. calibration / coherence tax
          - lambda_3 * constraint_violation_rate      # e.g. a safety / fairness leak
          - lambda_4 * max(0, over_correction_rate)   # e.g. selectivity / over-refusal
          - lambda_5 * max(0, leading_indicator)      # a cheap off-manifold early-warning axis
```

Design principles:

- **`max(0, ·)` on every cost term.** You penalize regressions, not improvements
  — a method is never rewarded for over-shooting a cost axis in the "good"
  direction, and never penalized for leaving it unchanged.
- **The penalties must dominate a degenerate win.** A method that produces
  garbage might trivially satisfy one constraint while destroying coherence; the
  coherence penalty must be large enough that the garbage method *cannot* top the
  table. Test this explicitly: construct a degenerate row and confirm it loses.
- **Weights `lambda_*` are pinned and SHA-256 fingerprinted.** The fingerprint
  appears in every reasoning entry and every dashboard footer. **Editing the
  formula mid-project to crown a favoured row is a BLOCKER** — it silently
  invalidates every prior comparison.
- **Always report the composite to 4 dp AND every axis separately.** Never
  collapse to one number in prose without the per-axis breakdown.
- **L1 — Validate the primary efficacy axis before optimizing it.** The
  `primary_efficacy` term is the engine of every keep/discard decision. Before
  the first hill-climb on a new topic or behavior, validate that the metric
  correctly measures the real phenomenon by computing its correlation/agreement
  with a trustworthy reference signal (human labels or an independent strong
  judge) on a calibration set. An endogenous proxy (one whose scoring logic
  was derived from the same source as the intervention) can score an opposite
  intervention as a success — this invalidates the entire optimization. See
  [`autoresearch-experiment`](../autoresearch-experiment/SKILL.md) §L6 for
  the domain-agnostic controls protocol.
- **L5 — The Pareto frontier is the primary scientific object.** The scalar
  composite is a convenience tiebreak. The core accept criterion is
  Pareto-dominance: a new config is accepted iff it is at least as good on
  every axis and strictly better on at least one. See
  [`autoresearch-per-hypothesis-hillclimb`](../autoresearch-per-hypothesis-hillclimb/SKILL.md)
  §L5 for the full protocol.

> *Worked example (image-classification).* `primary_efficacy` = held-out
> accuracy; cost axes = `accuracy_drop_on_clean`, `ECE_increase`,
> `worst-group_error` (fairness/constraint), `over-smoothing_rate`, and a cheap
> `feature-rank_collapse` leading indicator. A prior that lifts rare-class
> accuracy but blows up calibration cannot win: the ECE penalty prices it out.

> *Worked example (tabular-ML).* `primary_efficacy` = AUC; cost axes =
> `id/ood AUC gap`, `ECE`, `n_negative_folds`, `feature-leakage_score`. The
> leakage term is one-sided and large — it makes a leaked-feature "win"
> impossible to top the table with.

Owned conceptually here; enforced wherever the project pins
`COMPOSITE_FORMULA`. See also the rigor floor (§5).

---

## 5. The screening → hill-climb → evaluation funnel (+ the rigor floor)

A hypothesis travels through three tiers. **Confusing the tiers is the single
most common way an autoresearch program fools itself.**

```
   register          screen (n<=3)            hill-climb              evaluate (n>=7)
  hypothesis  ----->  1 config at the   ----->  20-25-trial    ----->  n>=7-seed confirm
  in IDEA_TABLE       documented base          coordinate              + rigor contract
                      (cheap, many)            descent over            + ordinal gate
                                               the tuning cube         -> EXTERNAL-READY
```

1. **Screen** one config per hypothesis at the documented baseline. Cheap; run
   many. **n<=3 seeds is SCREENING, full stop** — n=3 cannot reach p<0.05 under a
   paired test. A negative screening result is a statement about *the default
   config*, not about the prior itself.
2. **Hill-climb** a surfaced candidate via **coordinate descent over the tuning
   cube** — the topic's analogue of `(lr × wd × batch × optimizer × seed)` —
   20–25 trials, **strict-`>` champion rule** (a tied later candidate does NOT
   replace the champion; this protects against seed noise). Produces a
   per-hypothesis dashboard.
3. **Confirm** the hill-climbed best at **n>=7 seeds** and apply the rigor
   contract before any external claim.

**The statistical rigor floor.** Any sentence using **"winner" / "beats
baseline" / "outside seed noise" / "statistically significant"** binds this
four-part contract:

1. **Paired Wilcoxon signed-rank** test across matched seeds/folds.
2. **95% bootstrap CI** on the delta (≥10k resamples).
3. **Holm-Bonferroni** correction across the whole sweep family (you ran many
   configs; correct for it).
4. An **empirically-derived per-setting noise band** (2σ from same-config
   different-seed runs — measured, never a rule-of-thumb).

Plus the process gates:

- **Pre-register** the screening-vs-evaluation classification and the success
  criterion in version control **BEFORE** the sweep. Reclassifying a loser as
  "screening" after the fact is **HARKing** — a BLOCKER.
- **Ordinal gate.** A claim is `EXTERNAL-READY` only when the **worst**
  evaluation seed beats the **best** baseline seed, AND the four-part contract
  holds.
- **Verdict tiers** for hypotheses: `NOVEL+TESTABLE`, `DERIVATIVE+TESTABLE`,
  `NUMEROLOGY` (the result is real but *any nearby value of the knob would do* —
  the prior added nothing specific), `UNFALSIFIABLE`, `FALSIFIED`, and
  `UNTESTED_ON_RIGHT_DATASET` (the method has never been run on the dataset the
  claim is about — a verdict that blocks the claim until corrected).

Never hill-climb a BROKEN implementation (fix first) or a NUMEROLOGY hypothesis.

Owned by: [`../autoresearch-per-hypothesis-hillclimb/SKILL.md`](../autoresearch-per-hypothesis-hillclimb/SKILL.md)
(funnel) and [`../autoresearch-paper-rigor/SKILL.md`](../autoresearch-paper-rigor/SKILL.md)
(rigor floor + verdict tiers).

---

## 6. The orthogonal-axis combo ladder

Once individual priors are hill-climbed, you may stack them — but only under a
strict discipline:

- **Stack ONLY priors on orthogonal axes / disjoint intervention sites.** The
  decision rule: different site ⇒ stack; same site + same direction + different
  operation ⇒ they **compete** (do not stack); near-orthogonal directions ⇒ stack
  until the topic's "budget" axis (the cost term that accumulates) is spent.
- **Build an additive 2→N ladder:** each row adds **exactly ONE** new orthogonal
  prior on top of the previous row, so the marginal effect of each prior is
  readable.
- **The "everything-on" hybrid is FORBIDDEN.** Turning every prior on at once
  produces an unreadable result and, empirically, usually *underperforms* a
  curated additive stack (interactions and budget exhaustion). The hybrid is the
  classic way a combo campaign destroys its own attribution.
- **Hill-climb each prior at the single-prior level FIRST**, then combo. Do not
  combo-stack untuned priors.

> *Worked example.* Row 1: equivariant first block (tuned). Row 2: + a
> feature-decorrelation regularizer (a *different* site — the loss, not the
> filters — so orthogonal: stack). Row 3: + test-time augmentation (orthogonal
> again). You do NOT also add a *second* filter-init prior at the same first
> block — that is the same site, same direction; it competes, not stacks.

Owned by: [`../autoresearch-combo-ladder/SKILL.md`](../autoresearch-combo-ladder/SKILL.md).

---

## 7. The dashboard mandate (transparency is the product)

The dashboard is the deliverable; the weights are secondary. It must be **richly
detailed, fully transparent, self-contained, and HIERARCHICALLY LINKED** — a
master dashboard that drills down into per-hypothesis sub-dashboards, which drill
down into per-experiment leaf pages. **Sub-links to the sub-dashboards are
mandatory** — a master dashboard whose rows do not link down to their
per-hypothesis and per-experiment pages is incomplete and fails the audit.

```
            dashboard/index.html  (MASTER)
                     |
        +------------+------------+
        |            |            |   <-- every row sub-links down (mandatory)
        v            v            v
  ideas/01/        ideas/02/    ideas/NN/      (PER-HYPOTHESIS sub-dashboards)
  dashboard/       dashboard/   dashboard/
        |
        +--------> experiments/expNNN.html      (PER-EXPERIMENT leaf pages)
```

**A. Master dashboard** (`dashboard/index.html`, mirrored to the docs site):

- Sortable / type-to-filter runs table; default sort = `composite` desc; the
  global champion row highlighted. **Every numeric cell carries `n=X` + a
  `SCREENING`/`EVALUATION` tier chip** — no bare numbers.
- A **multi-axis radar / parallel-coordinates** panel per method (the
  multi-objective view — every cost axis visible at once).
- A **Pareto panel**: efficacy vs each cost axis, with prior methods as stars; at
  least one dominated row must exist (it proves the harness discriminates).
- The **ladder board**: per method, the rung reached + the gate cleared/failed +
  the `failure_reason`.
- The **stack/compete matrix** (the §6 decision rule rendered live from data).
- A 4-bullet "how to read this" orientation block; the `COMPOSITE_FORMULA`
  fingerprint + commit SHA in the footer.
- **Sub-links** from every row to its per-hypothesis and per-experiment pages.

**B. Per-hypothesis sub-dashboard** (`ideas/<NN>/dashboard/index.html`):

- Best-config callout; per-axis coordinate-descent Pareto small-multiples;
  seed-stability bars; a cells table linking to per-experiment pages; the
  hypothesis statement, its falsifier, the predicted Δ, and the current verdict;
  a back-link to the master.

**C. Per-experiment leaf page** (`.../experiments/expNNN.html`):

- The full 7-step reasoning entry (diagnosis, citations, hypothesis, prediction,
  verdict, learning) rendered from markdown; the sweep curves; the
  generation/output samples (intervened vs baseline) side by side; the cheap
  leading-indicator probes; all axis metrics with CIs.

**Hard rendering rules:** self-contained HTML (no CDN / JS framework; one inline
`<script>` for sort/filter); raster images (PNG) not SVG for dense plots;
markdown actually rendered (the rendering audit asserts no literal `##` / `**` /
`|---|` leaks into the page); absolute repository-blob links HEAD-tested;
small-multiples over dense charts; **no self-graded ACCEPT banner** without the
"Internal QA pass — independent external review pending" qualifier; no emoji
unless asked.

Owned by: [`../autoresearch-dashboard/SKILL.md`](../autoresearch-dashboard/SKILL.md),
[`../autoresearch-dashboard-comprehension/SKILL.md`](../autoresearch-dashboard-comprehension/SKILL.md),
[`../autoresearch-per-experiment-page/SKILL.md`](../autoresearch-per-experiment-page/SKILL.md),
[`../autoresearch-typography-and-rendering/SKILL.md`](../autoresearch-typography-and-rendering/SKILL.md),
[`../autoresearch-link-discipline/SKILL.md`](../autoresearch-link-discipline/SKILL.md).

---

## 8. Agent-team discipline

Compute (the GPU sweep) is usually **sequential**. But **docs / code / research /
audit / critique parallelize**. When you fan out:

- **Disjoint file scopes.** Each agent owns a non-overlapping set of paths. Two
  agents must never write the same file.
- **Scoped `git add <paths>`, never `git add -A`.** An agent commits only its own
  scope so a concurrent agent's half-finished edit is not swept into the commit.
- **Retry-wrapped commits.** 5 attempts with a pull-rebase fallback (parallel
  agents race on the same branch).
- **Bounded structured returns** (≤250 words) so the orchestrator can read every
  agent's result.
- **No self-grading circularity.** When the implementer, the impl-critic, and the
  sci-critic share a model family, every internal audit verdict MUST carry the
  disclosure: **"Internal QA pass — independent external review pending."** A
  same-family audit is a useful filter, never an external seal of approval.

Owned by: [`../autoresearch-multi-agent-dispatch/SKILL.md`](../autoresearch-multi-agent-dispatch/SKILL.md),
[`../autoresearch-critic-team/SKILL.md`](../autoresearch-critic-team/SKILL.md),
[`../autoresearch-scicritic-team/SKILL.md`](../autoresearch-scicritic-team/SKILL.md),
[`../autoresearch-fixer-campaign/SKILL.md`](../autoresearch-fixer-campaign/SKILL.md).

---

## 9. The axis taxonomy (what makes "one change" well-defined)

The core invariant says "change ONE thing". That is only meaningful if you have
enumerated the **orthogonal axes** of your topic up front. The taxonomy is the
first artifact you write (instantiation step 1, §11). It serves three jobs:

1. It defines the unit of a single-axis perturbation (§1).
2. It tells the combo ladder which axes are safe to stack (§6).
3. It scopes the hill-climb cube (§5).

A good taxonomy lists each axis, its admissible values, and the *mechanism* by
which it acts — so a diagnosis can name the specific axis that is the bottleneck.

> *Worked example.* For image-classification inductive biases the axes might be:
> {filter-init basis, equivariance group, normalization scheme, augmentation
> family, regularizer site, capacity/width}. "Add equivariance" and "change the
> regularizer site" are different axes ⇒ stackable. "Filter-init basis A" vs
> "filter-init basis B" are the same axis ⇒ they compete, not stack.

---

## 10. The dual-track audit (before any external claim)

No external claim ships without a **dual-track audit plus two negative controls**:

1. **Implementation-critic** ([`../autoresearch-critic-team/SKILL.md`](../autoresearch-critic-team/SKILL.md)):
   does the code actually do what the reasoning entry claims? Verdicts include
   `BROKEN` (→ Fixer campaign, then re-screen + re-hill-climb).
2. **Science-critic** ([`../autoresearch-scicritic-team/SKILL.md`](../autoresearch-scicritic-team/SKILL.md)):
   is the *claim* sound? Is it NUMEROLOGY, UNFALSIFIABLE, or HARKed? A
   hill-climbed winner with a NUMEROLOGY sci-verdict is NOT external-ready.
3. **Data-split audit** ([`../autoresearch-data-split-audit/SKILL.md`](../autoresearch-data-split-audit/SKILL.md)):
   the `audit_or_die()` leakage check — train/val/test disjoint, no target
   leakage, no duplicate rows across splits — run BEFORE any model build on a new
   dataset.
4. **Shuffle test** ([`../autoresearch-shuffle-test/SKILL.md`](../autoresearch-shuffle-test/SKILL.md)):
   the negative control — the method must NOT "work" when the labels/conditions
   are shuffled. A method that still wins on shuffled labels is measuring an
   artifact.

All four carry the same-model-family circularity disclosure (§8) when applicable.

---

## 11. How to instantiate this meta-process on a new topic (10 steps)

Run these in order. Each produces a concrete artifact; do not start step *k+1*
until step *k*'s artifact exists.

1. **Define the axes.** Write the orthogonal-axis taxonomy (§9): each axis, its
   admissible values, its mechanism. Artifact: `AXIS_TAXONOMY.md`.
2. **Define the composite.** Pick the primary efficacy axis and every cost axis;
   write the one-sided penalty formula (§4); pin and SHA-256 fingerprint it.
   Construct a degenerate row and confirm it loses. Artifact:
   `COMPOSITE_FORMULA` + its fingerprint.
3. **Wire datasets per rung.** Pin the UNIT/SMOKE/DEV/STANDARD/FULL data and eval
   subsets (§3); fixed seeds, pinned subsets so SMOKE is comparable across
   iterations. Artifact: the per-rung dataset registry. Run the data-split audit
   (§10) on each.
4. **Scaffold ideas/_TEMPLATE.** Create the per-hypothesis sub-project template
   (config, reasoning, experiments/, dashboard/) via
   [`../autoresearch-idea-scaffold/SKILL.md`](../autoresearch-idea-scaffold/SKILL.md).
5. **Write the runner contract.** The runner takes a config + a validated
   reasoning entry, runs at a specified rung, logs all axes, and **refuses to
   launch on an invalid/placeholder pre-run entry**. It calls `audit_or_die()`
   before any model build. Artifact: the runner module + its unit tests.
6. **Rung-0 tests.** Write UNIT tests: shape/contract, every Boolean-flag combo,
   the placeholder-refusal, exact state restoration. All green before any compute
   launches (the test discipline, README).
7. **Register hypotheses in IDEA_TABLE.** Enumerate the hypotheses with their
   falsifiers, predicted Δ, and screening/evaluation pre-classification (§5).
   Artifact: `IDEA_TABLE.md`.
8. **Screen.** One config per hypothesis at the documented baseline, n<=3 (§5).
   Append-only log; commit per experiment.
9. **Hill-climb** each surfaced candidate over the tuning cube; produce the
   per-hypothesis dashboard (§5).
10. **Evaluate + dashboard + audit.** Confirm at n>=7 with the rigor contract
    (§5); regenerate the master + sub-dashboards (§7); run the dual-track audit +
    both negative controls (§10) before any external claim. Archive a new global
    best with [`../autoresearch-winner-archive/SKILL.md`](../autoresearch-winner-archive/SKILL.md).

Crash safety wraps all ten: the checkpoint heartbeat
([`../autoresearch-checkpoint/SKILL.md`](../autoresearch-checkpoint/SKILL.md))
and session-resume
([`../autoresearch-session-resume/SKILL.md`](../autoresearch-session-resume/SKILL.md))
run continuously so a power outage never loses progress.

---

## 12. State files (the ledger)

| file | role |
|---|---|
| `experiment_log.jsonl` | append-only experiment history |
| `best_config.json` | global champion config + full results |
| `reasoning_annotations.json` | per-experiment 7-step entries (pre + post) |
| `running.json` | transient signal while an experiment runs |
| `AXIS_TAXONOMY.md` | the orthogonal-axis enumeration (§9) |
| `IDEA_TABLE.md` | the hypothesis registry + status + pre-classification |
| `EXPERIMENT_LEDGER.md` | promotion/demotion log (method · rung · axes · verdict) |
| `FINDINGS.md` | external-ready findings (rigor-gated only) |
| `ideas/<NN>/...` | per-hypothesis sub-project (idea-scaffold layout) |
| `dashboard/` + docs mirror | the rich multi-page dashboard |
| `audits/` | impl-critic, sci-critic, data-split, shuffle-test, meta-process audits |

---

## 13. Definition of done / self-audit checklist

The auditor checks that **every section maps to a concrete artifact**. The
process is "done" for a given claim only when all of the following hold:

- [ ] **Core invariant (§1):** every experiment in the log is a single-axis
      perturbation of the champion or a documented baseline. No multi-axis steps.
- [ ] **7-step ritual (§2):** every experiment has a pre-run entry that *passed
      the gates* (no placeholders) and a post-run verdict + learning. Citations
      are in full format with real identifiers.
- [ ] **Ladder (§3):** every method's highest rung is logged with the gate it
      cleared/failed; no method consumed rung *k+1* compute without clearing rung
      *k*. The same axes are measured at every rung.
- [ ] **Composite (§4):** the formula is pinned and fingerprinted; the fingerprint
      appears in every reasoning entry and dashboard footer; a degenerate row
      provably loses; the formula was not edited mid-project.
- [ ] **Funnel + rigor (§5):** screening (n<=3) and evaluation (n>=7) are
      pre-registered and not retro-reclassified; every "winner"/"beats baseline"
      sentence binds the four-part contract; the ordinal gate holds; the verdict
      tier is assigned.
- [ ] **Combo ladder (§6):** stacks are additive and orthogonal; no
      "everything-on" hybrid; each prior was hill-climbed before stacking.
- [ ] **Dashboard (§7):** master → per-hypothesis → per-experiment links all
      resolve; every numeric cell has `n=` + tier chip; the rendering audit finds
      no literal markdown leak; the "how to read this" block is present; the
      newcomer grounding block (research goal, domain primitives, open methodology
      tabs) is present; every table has a "What is this" glossary block; color
      legend present on all verdict-bearing tables.
- [ ] **Goal clarity:** README, dashboard, and paper each state the concrete
      outcome (what is being maximized, what SOTA means, what a winner is) — not
      just "this is a research program."
- [ ] **FINDINGS.md self-contained:** preamble with glossary present; every
      identifier defined before use; summary table present; strongest result
      stated plainly; zero-cross-reference rule holds.
- [ ] **EXPERIMENT_LEDGER.md interpretable:** "how to read this ledger" section
      present; column glossary complete; verdict-tier meanings defined; campaign-
      arc narrative updated for the current phase.
- [ ] **Provenance per tested hypothesis:** every tested idea has a provenance
      artifact (experiment tags, commands, artifact paths, result interpretation);
      untested ideas declare what new harness code they need; shared-harness
      architecture disclosed in README and dashboard if applicable.
- [ ] **Agent discipline (§8):** disjoint scopes, scoped adds, retry-wrapped
      commits; every internal audit carries the circularity disclosure.
- [ ] **Dual-track audit (§10):** impl-critic + sci-critic + data-split +
      shuffle-test all ran and passed (or their failures are logged and
      addressed) before the external claim.
- [ ] **Checkpoint (§11 wrap):** the working tree is clean; the session-resume
      document is current; nothing is at risk to a crash.
- [ ] **Objective validity (L1):** the primary efficacy metric has been validated
      against a trustworthy reference (correlation ≥ threshold on a calibration
      set) before the first hill-climb. Calibration result logged.
- [ ] **Mandatory controls (L6):** every effect-claim experiment row in the ledger
      includes delta_vs_random_magnitude and delta_vs_inverted_signal; both > 0
      for any kept result.
- [ ] **Extraction stability (L10):** every extracted direction used downstream
      has stability_p5 ≥ 0.85 (100-bootstrap cosine stability); logged in ledger.
- [ ] **Adaptive seeds (L3):** no champion update was made at n=1; confirmation
      follows the n=1 → n=3 → n=7 escalation.
- [ ] **Interaction surface (L4):** if coordinate descent oscillated, a joint
      surface was fitted and the champion derived from it.
- [ ] **Pareto-dominance gate (L5):** each champion update was verified to
      Pareto-dominate the prior champion on all cost axes, not just composite.
- [ ] **Scale tagging (L7):** every finding is tagged PROVISIONAL / SUPPORTED /
      SCALE-LIMITED with the model/setting coverage stated.
- [ ] **Confirmation holdout + promotion budget (L8):** confirmation set
      composition and K pre-registered in version control before screening;
      integrity maintained (no use during screening or hill-climb).
- [ ] **Power adequacy (L9):** minimum delta of interest pre-registered; n chosen
      to provide ≥ 80% power for that delta; result classified as SIGNIFICANT /
      INCONCLUSIVE / UNDERPOWERED.
- [ ] **Real external benchmark (L12):** behavior-efficacy claims at Rung 2+ are
      evaluated on a published benchmark with a population of items — not on
      researcher-authored synthetic data. Dataset name logged in every experiment.
- [ ] **Sign-aware verdict (L11):** every "win" / "supported" verdict verified to
      have Δ > 0 (not just |Δ| > 0 or p < α). Verdict logic gates on sign.
- [ ] **Item as replication unit (L13):** n_items and n_seeds reported separately;
      the item is the replication unit for the paired test when the benchmark
      provides a population; seeds measure within-item variance only.
- [ ] **Judge validated against benchmark ground truth (L14):** ROC-AUC on
      labeled benchmark items computed and disclosed before the first DEV-rung run.
      Judge fallback chain documented: Tier-A (API) → Tier-B (local off-family) →
      Tier-C (screening proxy, never backs a claim). Hard-abort if no valid judge.
- [ ] **Batching (L15):** generation and judging batched over all items in single
      GPU forward passes for any sweep over ≥ 50 items.
- [ ] **Two-model memory budget (L16):** models loaded sequentially (generate →
      unload → judge); pre-flight VRAM/RAM check before any two-model run.
- [ ] **Honest scoreboard (L12):** when prior INTERNAL claims are re-evaluated on
      a real benchmark, an explicit scoreboard of generalizing vs non-generalizing
      claims, with a synthesis sentence, is in FINDINGS.md. Null results have equal
      visual prominence to positive results.
- [ ] **Plain-English accessibility (L14, doc-organization):** every hypothesis/
      experiment page opens with an "In Plain English" box defining all technical
      terms in-page; project GLOSSARY exists and is linked from all pages.

A claim that cannot tick every relevant box is `INTERNAL` only and ships with the
"independent external review pending" qualifier.

---

## Anti-patterns

| Anti-pattern | Do instead |
|---|---|
| "Try X and see" | "X because [diagnosis] + [paper] predicts [mechanism]" |
| Grid search over a knob | Diagnose → hypothesize → test ONE value with justification |
| Change 2+ axes at once | ONE axis. Sequence them if both matter. |
| Report the aggregate only | Always read the per-axis + per-rung breakdown |
| Repeat a failed axis | 3 failures on one axis ⇒ rethink the diagnosis, not the value |
| Edit the composite to crown a row | The formula is fingerprinted; editing it is a BLOCKER |
| Win a constraint via a degenerate output | The coherence/cost penalty must price it out — test that it does |
| Call an n=3 result a "winner" | n<=3 is SCREENING; "winner" needs n>=7 + the four-part contract |
| Reclassify a loser as "screening" after the fact | HARKing — pre-register the classification before the sweep |
| "Everything-on" hybrid stack | Orthogonal-axis additive ladder only (§6) |
| Hill-climb a BROKEN implementation | Fix first (Fixer campaign), then re-screen + re-hill-climb |
| Master dashboard with no sub-links | Sub-links to per-hypothesis + per-experiment pages are mandatory |
| Self-graded ACCEPT banner | Carry the "independent external review pending" disclosure |
| Run the FULL benchmark to find a plumbing bug | Catch it at UNIT/SMOKE; climb the ladder |
| Skip the negative control | A method that wins on shuffled labels is measuring an artifact |
| Hill-climb before validating the objective metric | Validate the primary metric against a reference first (ρ ≥ threshold); see L1 in [`autoresearch-experiment`](../autoresearch-experiment/SKILL.md) |
| Champion update at n=1 | No champion decision without ≥ n=3 confirmation; see L3 in [`autoresearch-per-hypothesis-hillclimb`](../autoresearch-per-hypothesis-hillclimb/SKILL.md) |
| Coordinate descent when axes interact | Detect interaction → fit joint surface → resume CD; see L4 |
| Scalar-only ranking for Pareto-multiobjective problems | Use Pareto dominance as the primary accept criterion; composite is a tiebreak only; see L5 |
| Claim directional effect without random + inverted controls | Both controls mandatory; delta over null is not sufficient; see L6 in [`autoresearch-experiment`](../autoresearch-experiment/SKILL.md) |
| External claim from single-scale results | Tag PROVISIONAL until cross-scale replication; see L7 in [`autoresearch-paper-rigor`](../autoresearch-paper-rigor/SKILL.md) |
| No program-level confirmation holdout | Pre-designate and never-touch confirmation set + promotion budget K; see L8 |
| n=7 assumed adequate for all effect sizes | Pre-register minimum delta; compute required n; see L9 |
| Use an extracted direction without checking stability | Bootstrap-cosine stability_p5 ≥ 0.85 gate; see L10 in [`autoresearch-experiment`](../autoresearch-experiment/SKILL.md) |
| Claiming a behavioral result on researcher-authored synthetic data | Synthetic is SCREENING only; real published benchmark with item population required; see L12 in [`autoresearch-paper-rigor`](../autoresearch-paper-rigor/SKILL.md) |
| Significant result labelled "win" without checking sign | |Δ| > 0 is not sufficient; gate on Δ > 0; see L11 in [`autoresearch-paper-rigor`](../autoresearch-paper-rigor/SKILL.md) |
| "n=7 replications" when n=7 refers to seeds on one item | Report n_items and n_seeds separately; item is the replication unit; see L13 |
| No judge ROC-AUC validation against benchmark ground truth | Judge discrimination unknown; must disclose AUC before any DEV+ run; see L14 in [`autoresearch-paper-rigor`](../autoresearch-paper-rigor/SKILL.md) |
| Silent fallback to endogenous proxy when judge unavailable | Hard-abort required; proxy may only back SCREENING with explicit tag; see judge fallback chain |
| Per-item generation/judge loops over large benchmarks | 10–16x slower than batching; makes sweeps infeasible; batch all items |
| Both models in VRAM simultaneously on constrained hardware | OOM or commit-limit crash; load sequentially: generate → unload → judge |
| Re-evaluating on real benchmark without honest scoreboard | Reader cannot distinguish which prior claims survived; null results must have equal prominence |
| Technical-only documentation with no plain-English box | Non-specialist readers cannot engage; "In Plain English" box required on every hypothesis/experiment page; see L14 in [`autoresearch-doc-organization`](../autoresearch-doc-organization/SKILL.md) |

---

## Cross-references

- One experiment: [`../autoresearch-experiment/SKILL.md`](../autoresearch-experiment/SKILL.md)
- The ladder: [`../autoresearch-tiered-ladder/SKILL.md`](../autoresearch-tiered-ladder/SKILL.md)
- The funnel / hill-climb: [`../autoresearch-per-hypothesis-hillclimb/SKILL.md`](../autoresearch-per-hypothesis-hillclimb/SKILL.md)
- Rigor floor + verdict tiers: [`../autoresearch-paper-rigor/SKILL.md`](../autoresearch-paper-rigor/SKILL.md)
- Findings + ledger discipline: [`../autoresearch-findings-ledger/SKILL.md`](../autoresearch-findings-ledger/SKILL.md)
- Negative control: [`../autoresearch-shuffle-test/SKILL.md`](../autoresearch-shuffle-test/SKILL.md)
- Leakage audit: [`../autoresearch-data-split-audit/SKILL.md`](../autoresearch-data-split-audit/SKILL.md)
- Master dashboard: [`../autoresearch-dashboard/SKILL.md`](../autoresearch-dashboard/SKILL.md)
- Reader's guide discipline: [`../autoresearch-dashboard-comprehension/SKILL.md`](../autoresearch-dashboard-comprehension/SKILL.md)
- Leaf pages: [`../autoresearch-per-experiment-page/SKILL.md`](../autoresearch-per-experiment-page/SKILL.md)
- Rendering rules: [`../autoresearch-typography-and-rendering/SKILL.md`](../autoresearch-typography-and-rendering/SKILL.md)
- Link discipline: [`../autoresearch-link-discipline/SKILL.md`](../autoresearch-link-discipline/SKILL.md)
- Impl-critic: [`../autoresearch-critic-team/SKILL.md`](../autoresearch-critic-team/SKILL.md)
- Sci-critic: [`../autoresearch-scicritic-team/SKILL.md`](../autoresearch-scicritic-team/SKILL.md)
- Fixer campaign: [`../autoresearch-fixer-campaign/SKILL.md`](../autoresearch-fixer-campaign/SKILL.md)
- Agent dispatch: [`../autoresearch-multi-agent-dispatch/SKILL.md`](../autoresearch-multi-agent-dispatch/SKILL.md)
- Checkpoint heartbeat: [`../autoresearch-checkpoint/SKILL.md`](../autoresearch-checkpoint/SKILL.md)
- Session resume: [`../autoresearch-session-resume/SKILL.md`](../autoresearch-session-resume/SKILL.md)
- Idea scaffold: [`../autoresearch-idea-scaffold/SKILL.md`](../autoresearch-idea-scaffold/SKILL.md)
- Combo ladder: [`../autoresearch-combo-ladder/SKILL.md`](../autoresearch-combo-ladder/SKILL.md)
- Winner archive: [`../autoresearch-winner-archive/SKILL.md`](../autoresearch-winner-archive/SKILL.md)
- The pack index: [`../README.md`](../README.md)

---

## Closing reminder

The autoresearch loop is not "more sweep". It is the discipline that
distinguishes *"this knob helped at one config"* from *"this prior helps when it
has been given a fair shake, prices every objective axis, climbs the full ladder,
survives the negative controls, and an independent reviewer could reproduce it."*
Start from the champion, change one thing, price everything, climb the ladder,
prove it outside the noise, render it transparently, and never lose a result to a
crash. Everything else in this pack is the *how* of one of those clauses.
