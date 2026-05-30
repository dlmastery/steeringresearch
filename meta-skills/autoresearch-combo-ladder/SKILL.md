---
name: autoresearch-combo-ladder
description: Use when designing multi-prior hybrid experiments after individual priors have been hill-climbed. Stack ONLY priors that touch orthogonal (non-competing) intervention sites. Build an additive 2→N ladder where each row adds exactly ONE new orthogonal prior — this makes the marginal effect of each axis readable. The "everything-on" hybrid is forbidden.
---

# Skill — Orthogonal-axis additive combo ladder

## When to use

- After a single-prior hill-climb has produced at least one confirmed
  positive result and you want to test whether multiple priors compound.
- When the user asks about "stacked experiments", "combo hybrids",
  "combining priors", or whether design choices interact.
- When designing the sweep matrix for a multi-prior campaign.
- NEVER before each individual prior has been hill-climbed at the
  single-prior level (see Anti-patterns). Combo-stacking untuned
  priors conflates axis interactions with configuration deficit.

## The stack-only-orthogonal-axes rule

Two priors are **orthogonal** (stackable) if they act on different,
logically independent intervention sites in the pipeline. Two priors
**compete** (not stackable) if they act on the same site, the same
direction, or are mutually exclusive alternatives:

| relationship | action |
|---|---|
| Different intervention site | **Stack** — one row adds this prior |
| Same site + same direction + different operation | **Compete** — pick one; do not stack |
| Near-orthogonal but share a budget axis | Stack until the budget axis is spent; stop there |
| Mutually exclusive alternatives at a slot | ONE prior per slot; the slot takes the better of the two |

The project's axis taxonomy (see
[`../autoresearch-meta/SKILL.md`](../autoresearch-meta/SKILL.md) §9)
defines which axes are independent. A combo campaign is only valid
if the axis taxonomy was written BEFORE the campaign.

## The additive 2→N ladder

```
N=2: base + 1 orthogonal prior            (1 new axis added)
N=3: base + 2 orthogonal priors           (1 new axis added)
N=4: base + 3 orthogonal priors           (1 new axis added)
...
N=K: base + (K-1) priors, one per axis   (1 new axis added at each step)
```

Each row N adds **exactly ONE** new orthogonal prior on top of row N-1.
Reading `metric(row N) - metric(row N-1)` gives the marginal effect of
the newly-added axis. This is the property that makes the ladder
informative and attributable.

The base for row 2 is the tuned single-prior champion (not the
project-default recipe, not an untuned baseline). Each row inherits
the best config from its predecessor.

## The "everything-on" hybrid is forbidden

Turning every prior on at once produces an unreadable result and,
empirically, usually underperforms a curated additive stack because:

1. **Attribution is lost.** You cannot tell which prior contributed
   which portion of the gain or loss.
2. **Budget exhaustion.** Priors that each separately respect a cost
   budget (memory, parameter count, an objective-axis tolerance) often
   collectively violate it when combined, producing a net regression.
3. **Interaction confounding.** A negative interaction between two
   specific priors is invisible in a full-hybrid result and invisible
   in per-row attribution.

The only place a "full-hybrid" row is permitted is as the FINAL row in
the ladder (after the additive rows are complete), as a check on whether
the full combination retains the additive gains. Even then it is a
diagnostic row, not the headline row.

## Hill-climb-each-prior-BEFORE-combo

Each prior must have been hill-climbed at the single-prior level before
it enters the combo ladder. The reason: a prior that appears weak at
the default config may be weak because it is mis-tuned, not because it
is uninformative. Including a mis-tuned prior in a combo row conflates
"bad prior" with "bad config", destroying the row's attributability.

The pre-condition checklist for launching a combo campaign:

- [ ] Each candidate prior has a `hillclimb_results.json` with a
      confirmed best config (n ≥ 3 seeds at the hill-climbed best).
- [ ] Each prior has a `KEEP` or `NEAR-MISS` verdict from the screening
      and hill-climb phases (falsified and NUMEROLOGY priors are excluded).
- [ ] The project's axis taxonomy confirms all candidate priors act on
      distinct sites (no competing-site pairs in the ladder).
- [ ] The combo ladder rows are written into a pre-registered
      `combo_plan.yaml` BEFORE the first combo run.

## Sweep-row implementation pattern

```python
# Pseudocode — adapt to the project's config schema.
# Each step spreads the previous row's config and adds ONE new key.

prior_A_best = load_best_config("prior_A")   # from hillclimb_results.json
prior_B_best = load_best_config("prior_B")   # orthogonal to A
prior_C_best = load_best_config("prior_C")   # orthogonal to A and B

rows = [
    dict(tag="combo_2_A",
         overrides=dict(**prior_A_best)),

    dict(tag="combo_3_AB",
         overrides=dict(**prior_A_best,
                        **{k: v for k, v in prior_B_best.items()
                           if k not in prior_A_best})),   # add B's axes

    dict(tag="combo_4_ABC",
         overrides=dict(**prior_A_best,
                        **{k: v for k, v in prior_B_best.items()
                           if k not in prior_A_best},
                        **{k: v for k, v in prior_C_best.items()
                           if k not in prior_A_best
                           and k not in prior_B_best})),  # add C's axes

    # Diagnostic-only: full hybrid (all priors combined)
    dict(tag="combo_full_diagnostic",
         overrides=dict(**prior_A_best, **prior_B_best_unique,
                        **prior_C_best_unique)),
]
```

The additive structure must be visible in the code: each row spreads
the previous and adds ONE new scope of keys. Hard-coding the full
config for each row breaks the additive invariant.

## Selecting what NOT to stack

Exclude a prior from the combo ladder if:

- It was **falsified** (a confirmed negative at the single-prior level
  with the pre-registered falsifier triggered).
- It occupies a **competing slot** — the same intervention site as a
  prior already in the ladder (pick the stronger of the two, not both).
- It was flagged **BROKEN** by the implementation critic — fix it first,
  then re-screen and re-hill-climb before including.
- It was classified **NUMEROLOGY** by the science critic — the mechanism
  is coincidental; there is no reason to expect it to stack usefully.

## Output interpretation

After the ladder runs, read the marginal column:

| pattern | interpretation |
|---|---|
| Each row improves on the previous | Strict additivity: priors are genuinely complementary |
| Marginals decrease but stay positive | Sub-additivity: priors interact (shared dynamics or budget); the ladder reveals which step is the culprit |
| A specific row reverses the gain | The newly-added prior at that row interacts negatively with the accumulated stack; consider removing it |
| Super-additivity (row N > sum of single-prior lifts) | Rare; suggests genuine synergy — investigate the interaction mechanism before claiming it |

The headline of a combo campaign is the row with the highest composite
that can be attributed cleanly. That is usually NOT the final row if
the ladder turns at some point.

## Anti-patterns

| Anti-pattern | Why it fails |
|---|---|
| Turn everything on and report the result | Attribution is lost; empirically often underperforms the curated stack |
| Stack priors that compete at the same site | The later prior overrides or interferes with the earlier; result is not additive |
| Stack an untuned prior | Conflates config deficit with axis interaction |
| Include a falsified or NUMEROLOGY prior | Wastes compute; confounds the stack's attributability |
| Two-prior compound without the ladder context | Cannot tell if the effect generalises across additional priors |
| Report the full-hybrid row as the headline | It is a diagnostic; the attribution-clean row is the headline |
| Build the ladder before the axis taxonomy exists | The "orthogonal" classification is undefined without the taxonomy |

## Cross-references

- [`../autoresearch-meta/SKILL.md`](../autoresearch-meta/SKILL.md) §6
  — the spine section that defines the orthogonal-axis combo rule and
  cross-references this skill as the owning workflow.
- [`../autoresearch-per-hypothesis-hillclimb/SKILL.md`](../autoresearch-per-hypothesis-hillclimb/SKILL.md)
  — the prerequisite: each prior must be hill-climbed before it enters
  the combo ladder.
- [`../autoresearch-ablation-sweep/SKILL.md`](../autoresearch-ablation-sweep/SKILL.md)
  — the screening counterpart; identifies which single priors are worth
  hill-climbing and therefore worth including in the combo ladder.
- [`../autoresearch-experiment/SKILL.md`](../autoresearch-experiment/SKILL.md)
  — each combo row is one experiment under the 7-step ritual.
- [`../autoresearch-dashboard/SKILL.md`](../autoresearch-dashboard/SKILL.md)
  — the stack/compete matrix panel on the master dashboard renders the
  combo ladder's orthogonality decision live from data.
- [`../autoresearch-tiered-ladder/SKILL.md`](../autoresearch-tiered-ladder/SKILL.md)
  — combo rows must also clear the rung promotion gates; a combo that
  clears SMOKE does not skip to FULL.
