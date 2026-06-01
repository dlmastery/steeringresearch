---
name: autoresearch-paper-rigor
description: Use BEFORE making any "winner" / "outside noise" / "statistically significant" claim in any paper, findings document, README, or dashboard headline. Enforces the statistical-rigor floor (paired Wilcoxon + 95% bootstrap CI + Holm-Bonferroni), the seed-count floor, pre-registration of the screening-vs-evaluation distinction, dataset-aware verdict tiers, the abstract-must-carry-limitations discipline, the internal-contradiction self-audit, and the auditor-circularity disclosure.
---

# Skill — Statistical rigor floor + pre-registration discipline

## When to use

- BEFORE writing any "winner", "outside seed noise", or
  "statistically significant" sentence in the paper, FINDINGS,
  README, or dashboard headline.
- BEFORE running the seed-escalation sweep — pre-register the success
  criterion in the version-control system.
- AFTER any audit produces a "WEAK_REJECT" verdict — downgrade the
  artefact within the same commit and re-issue the claim with the
  required level of rigor.
- BEFORE recording a NUMEROLOGY or FALSIFIED verdict on a hypothesis
  — confirm the dataset matches the pre-registered falsifier.

## Pillar 1 — The statistical-rigor floor

Every empirical claim that uses ANY of the trigger phrases below MUST
report all four items in the contract.

**Trigger phrases:** "winner", "outside seed noise", "statistically
significant", "beats baseline", "positive result", "lead", "lift",
"survives at α=...".

**Contract (mandatory, all four):**

1. **Paired Wilcoxon signed-rank** (or paired t-test with explicit
   normality justification — Shapiro-Wilk p-value required).
   - For 3 paired seeds: Wilcoxon W ∈ {0, 1, ..., 6}; the smallest
     achievable two-sided p is 0.25 (W=0 or W=6).
   - This means **n=3 cannot achieve p<0.05 under Wilcoxon** —
     n=3 is SCREENING, period.
   - For n=7 paired seeds: W=0 yields p=0.0156 two-sided, which
     clears Holm-Bonferroni α'=0.0167 for a 3-hypothesis family.

   ```python
   from scipy.stats import wilcoxon
   stat, p = wilcoxon(leader_seeds, baseline_seeds, alternative='greater')
   ```

2. **95% bootstrap CI on the pp delta** (>= 10,000 resamples,
   percentile method):
   ```python
   import numpy as np
   def bootstrap_ci(leader, baseline, n_boot=10_000, seed=0):
       rng = np.random.default_rng(seed)
       deltas = np.array(leader) - np.array(baseline)
       boots = rng.choice(deltas, size=(n_boot, len(deltas)),
                          replace=True).mean(axis=1)
       return np.percentile(boots, 2.5), np.percentile(boots, 97.5)
   ```

3. **Holm-Bonferroni correction** across the sweep family.
   - K = number of hypothesis-versus-baseline comparisons in the
     family.
   - α'_i = α / (K − i + 1) for the i-th smallest p, in ascending
     p order; reject if p_i ≤ α'_i.
   - Example: for a 3-winner family at α=0.05 — α'_1=0.0167,
     α'_2=0.025, α'_3=0.05.

4. **Empirically-derived noise band per split** — NOT a rule-of-thumb
   "±X%". Derive σ_seed from the project's own multi-seed baseline
   runs on the relevant held-out split and report:
   - A lift "outside seed noise" must exceed 2σ_seed, or be
     explicitly characterised at a smaller multiplier.
   - Cite the run paths (in `paper/STATISTICAL_TESTS.md` or
     equivalent) so the noise band is reproducible.

### Minimum-seed floor table

| family size K | α target | needed two-sided p | minimum n (paired Wilcoxon) |
|---:|---:|---:|---:|
| 1  | 0.05 | 0.05   | 6 (W=0 ⇒ p=0.0313) |
| 3  | 0.05 | 0.0167 | 7 (W=0 ⇒ p=0.0156) |
| 5  | 0.05 | 0.01   | 8 |
| 10 | 0.05 | 0.005  | 9 |

**n=3 is SCREENING** under any defensible test. **n>=7 is the
EVALUATION minimum** for a 3-hypothesis family at α=0.05 under
Holm-Bonferroni.

## Pillar 2 — Pre-registration of screening vs evaluation

The classification of any sweep row as SCREENING vs EVALUATION MUST be
pre-registered BEFORE the sweep runs. Operational pattern:

```yaml
# ideas/<NN>/experiments/exp<NNN>_<tag>/pre-registration.yaml
tag: <hypothesis_short_name>
classification: EVALUATION           # one of: SCREENING | EVALUATION
sweep_family: <family_label>         # K hypotheses sharing this family name
holm_bonferroni_k: <K>
alpha_target: 0.05
seeds_planned: [0, 1, 2, 3, 4, 5, 6]  # n=7 to clear α'=0.0167 for K=3
success_criterion: |
  Paired Wilcoxon W <= 1 vs <baseline_tag> on <held_out_split>;
  95% bootstrap CI lower bound > 0;
  Holm-Bonferroni α'_i at rank i.
falsifier_split: <split_name_or_dataset>
pre_registered_at: <ISO-8601 timestamp>
pre_registered_commit: <SHA>
author: <researcher_id>
```

The pre-registration commit SHA is referenced in the paper and in the
FINDINGS document when the claim is reported. Without pre-registration:

- Any reclassification of a row as "screening" after seeing it lose
  is **HARKing** (Hypothesizing After Results are Known) — a
  BLOCKER-level finding.
- Any redefinition of the success criterion after the sweep completes
  is HARKing.

### Distinguish ordinal-margin from Δmean

When reporting a lift, report BOTH statistics unambiguously:

> `<hypothesis>` lifts the ordinal margin
> `min(leader_seeds) − max(baseline_seeds) = +X.XX pp`
> AND the Δmean `mean(leader) − mean(baseline) = +Y.YY pp`.
> These are different statistics. The ordinal margin is a
> non-parametric gate; the Δmean is the sample mean of paired
> differences and is the basis of the Wilcoxon test. Report both;
> do not conflate them.

## Pillar 3 — Dataset-aware verdict tiers

A hypothesis whose pre-registered falsifier specifies a dataset or
split that is NOT in the sweep cannot earn a NUMEROLOGY or FALSIFIED
verdict from that sweep. The correct verdict is `UNTESTED_ON_RIGHT_DATASET`.
("split" and "dataset" are synonyms here — both refer to the evaluation
testbed the falsifier was pre-registered against.)

| verdict | meaning |
|---|---|
| NOVEL+TESTABLE | passed scientific scrutiny; awaiting empirical evidence |
| DERIVATIVE+TESTABLE | rediscovers a literature mechanism; lift is reproducible |
| NUMEROLOGY | passed audit but the mechanism is decorative coincidence |
| UNFALSIFIABLE | mechanism cannot be tested under any feasible protocol |
| FALSIFIED | pre-registered falsifier triggered |
| **UNTESTED_ON_RIGHT_DATASET** | falsifier specified split/dataset X; sweep ran on Y; verdict deferred |

The `UNTESTED_ON_RIGHT_DATASET` tier prevents a reviewer from labelling
a hypothesis NUMEROLOGY solely because the project happened to run on
the wrong testbed. The incorrect labelling is a BLOCKER-level finding.

## Pillar 4 — Limitations in the abstract; internal-contradiction audit

### Limitations IN THE ABSTRACT

Material limitations MUST appear in the abstract, not buried in a
limitations section. Specifically:

- Single-seed sweep rows: e.g., "the N-row development sweep is n=1
  per hypothesis."
- Baseline-below-SOTA gap: e.g., "our short-epoch baseline is X pp
  below the published SOTA."
- Auditor-self-grading circularity (see Pillar 5).
- Untested NOVEL+TESTABLE hypotheses: if a hypothesis is the sole
  novel survivor and has not been run on the designated evaluation
  split, say so.
- Wrong-testbed admission: if the primary evaluation dataset is not
  the correct testbed for the majority of the hypotheses tested, say
  so.

### Internal-contradiction self-audit

BEFORE marking the paper "done", read it end-to-end and verify:

1. The abstract's claim about contributions matches the conclusion's
   claim about contributions.
2. The introduction's scope qualifications are NOT contradicted by
   later sections marketing a broader claim.
3. The conclusion's "no NOVEL+TESTABLE-and-implementation-PASS
   hypothesis emerged" statement is NOT contradicted by abstract or
   marketing text that names a winner.
4. The "evaluation regime" framing is NOT contradicted by a caveat
   elsewhere identifying that regime as screening.
5. Every section heading or label that appears more than once
   (e.g., duplicated subsection numbers) is corrected.

If ANY contradiction exists, the paper is not done.

## Pillar 5 — Auditor-self-grading circularity disclosure

When any audit-derived rate (e.g., "X% non-PASS audit verdict") is
reported, include the calibration disclosure inline — in the section
that reports the rate, NOT in the limitations section:

> The implementer, critic, scientific-critic, and fixer agents
> share a model family. The reported non-PASS rate has not been
> calibrated against a known-good reference codebase. A future-work
> item is to re-run the audit protocol on an external reference
> implementation and report the non-PASS rate as a false-positive
> baseline. Until that calibration is reported, the reader should
> interpret the rate as descriptive, not diagnostic.

This disclosure is mandatory whenever the audit rate appears as
evidence in support of a claim about the project's hypotheses.

## Anti-patterns

- **Trigger phrases without the four-item contract.** "Outside seed
  noise" without paired Wilcoxon + bootstrap CI + Holm-Bonferroni is
  an unsubstantiated claim.
- **Reporting Δmean only when the ordinal margin tells a different
  story.** Always report both; do not cherry-pick the more favourable
  statistic.
- **Reclassifying a losing row as "screening" after the fact.** That
  is HARKing. BLOCKER-level. Pre-register or accept the negative
  result.
- **NUMEROLOGY verdict on a hypothesis whose pre-registered falsifier
  specified a different split or dataset.** Use
  `UNTESTED_ON_RIGHT_DATASET`.
- **Marketing the positive result without the calibrated limitations
  in the abstract.** Both must receive equal prominence.
- **n=3 "winner" reported without the SCREENING tag.** n=3 is
  screening under any defensible statistical test.
- **Placing the circularity disclosure only in a limitations
  appendix.** It must appear in the section where the audit rate is
  cited as evidence.

## Cross-references

- [`autoresearch-per-hypothesis-hillclimb`](../autoresearch-per-hypothesis-hillclimb/SKILL.md)
  — the n=3 → n>=7 escalation protocol. Hill-climb establishes the
  tuned ceiling; this skill establishes whether that ceiling is
  statistically distinguishable from the baseline.
- [`autoresearch-shuffle-test`](../autoresearch-shuffle-test/SKILL.md)
  — semantic leakage gate; a green shuffle test is a prerequisite for
  the rigor floor to be meaningful.
- [`autoresearch-data-split-audit`](../autoresearch-data-split-audit/SKILL.md)
  — structural split audit; both structural and semantic audits must
  be green before the rigor floor is invoked.
- [`autoresearch-tiered-ladder`](../autoresearch-tiered-ladder/SKILL.md)
  — the rung at which a result was produced determines the minimum
  seed count and the correct verdict tier.
- [`autoresearch-experiment`](../autoresearch-experiment/SKILL.md)
  — the single-experiment ritual that produces the raw metrics this
  skill's floor is applied to.
- Project-level `paper/STATISTICAL_TESTS.md` or equivalent — the
  pre-computed Wilcoxon / bootstrap / Holm-Bonferroni values for the
  project's current winner family.
- Project-level `paper/REVIEWER_CHECKLIST.md` or equivalent — the
  gating sections that consume this skill's outputs.
- [`autoresearch-findings-ledger`](../autoresearch-findings-ledger/SKILL.md)
  — the self-contained FINDINGS and interpretable ledger mandates; this
  skill's rigor floor determines which claims qualify for FINDINGS.md,
  and the findings-ledger skill governs how FINDINGS.md presents them.
