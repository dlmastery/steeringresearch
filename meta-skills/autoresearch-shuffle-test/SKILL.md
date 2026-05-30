---
name: autoresearch-shuffle-test
description: Semantic leakage detection by shuffling training targets and re-evaluating — if the model's validation score collapses to chance the train-to-val alignment is real; if it does not collapse, the evaluator is reading targets through a leakage channel. Catches the bug class the structural data-split audit cannot see: target-encoded features, label-dependent augmentation, inadvertent label echoes in the feature matrix, off-by-one alignment that lets the evaluator see a shifted version of the target.
---

# Skill — Shuffle-test audit (semantic leakage gate)

## When to use

- **Before any external claim** that names a model or method: a
  FINDINGS headline, paper abstract, README badge, or deployment
  decision. The dual-track audit (implementation-critic +
  scientific-critic + structural data-split + **shuffle test**) must
  have all four legs green. This skill is the fourth leg.
- **When alignment between inputs and targets is non-trivial** —
  groupby aggregations, walk-forward cross-validation, custom block
  splits, leave-one-group-out. These are the alignment bugs that the
  structural audit (`autoresearch-data-split-audit`) cannot detect.
- **When cross-validation uses custom split logic** that the project
  authored (not a vetted, widely-audited library splitter).
- **When the structural audit is green but the headline number is
  suspiciously strong** — a result that exceeds the established SOTA
  by more than 5% relative should automatically trigger a shuffle
  test before any claim is made.
- **After any change to the feature-engineering pipeline** that
  touches the target column: target encoding, leave-one-out encoding,
  k-fold target encoding, rank-encoding by target.
- **After a fixer-campaign patch** to any data-loader, augmenter, or
  evaluator module — the shuffle test is the re-smoke gate.

## Why

The structural data-split audit (`autoresearch-data-split-audit`)
catches the bug class **"a sample appears in both the training set and
the validation set"** — row-level overlap, group-level leakage, temporal
lookahead. It **cannot** catch the bug class **"the evaluator is reading
the target through a feature channel"**:

1. **Target column present in the feature matrix.** A column named
   `target_lagged` or `outcome_flag` was retained by accident.
   Structural audit sees zero row overlap and stamps green; validation
   metric is near-perfect.
2. **Group-aware target encoding computed on the full dataset** instead
   of the training partition only. Every validation sample's feature
   vector encodes that sample's own target. Validation is meaningless.
3. **Augmentation policy is label-conditional.** Samples from one class
   receive one transform, samples from another class receive a different
   transform, and the magnitude or type of the transform leaks the label.
4. **Off-by-one alignment between training and evaluator.** Training
   pairs `(x[i], y[i])` while the evaluator pairs `(x[i], y[i+1])`;
   on a dataset with autocorrelated targets, the evaluator is reading
   future information. This is the canonical time-series alignment bug:
   a large, apparently real uplift that collapses to near-zero after
   the fix.
5. **Inadvertent label echo in derived features.** A rolling-mean
   feature computed over a window that included the target row itself.

The shuffle test is the canonical semantic-leakage detector: **permute
the training labels, retrain from scratch, and evaluate on the
unshuffled validation set. The aggregate validation metric MUST collapse
to the chance baseline.** If it does not collapse, the evaluator is
reading the targets through a feature or alignment channel.

## The 3 shuffle modes

### Mode A — Hard shuffle (default)

Random permutation of all training targets with a fixed seed. Refit the
model from scratch. Evaluate on the **untouched** validation set.

**Expected outcome.** The aggregate validation metric drops to the chance
baseline:

- Binary classification: AUROC → 0.50 ± 0.02; AUPRC → positive class
  prevalence in the validation set.
- Multi-class classification: accuracy → 1/n_classes ± 0.02.
- Regression: R² → 0.0 ± 0.05; MSE → var(y_val).
- Ranking / trading: primary metric → 0.0 ± noise.

**Verdict.**
- **PASS** — shuffled metric within tolerance of the chance baseline.
  The original headline number reflects a real (train, val) alignment.
- **WEAK** — shuffled metric is > 2× tolerance distance from chance
  but < 5× tolerance distance. Investigate the highest-importance
  feature for partial label encoding.
- **FAIL** — shuffled metric is > 5× tolerance distance from chance
  OR is statistically indistinguishable from the unshuffled headline.
  STOP. The evaluator is reading the targets. Fix the alignment or
  remove the leaking feature and re-audit before any external claim.

### Mode B — Within-group shuffle (grouped CV)

For grouped or blocked CV (leave-one-group-out, leave-one-entity-out,
leave-one-subject-out): permute training targets **within each group**,
then retrain.

**Why this mode exists.** Hard-shuffle Mode A on grouped data may
artificially destroy class-group correlations and produce a false PASS.
Within-group shuffle preserves the group-level class prior and forces
the model to find within-group signal. If no within-group signal exists,
the within-group shuffled metric drops to chance.

**Expected outcome.** Same chance baselines and PASS / WEAK / FAIL
thresholds as Mode A.

### Mode C — Block shuffle (time-aware)

For time-series or walk-forward CV: permute training targets **within
each walk-forward fold**, then refit. Evaluate on the unshuffled future
fold(s).

**Why this mode exists.** Hard-shuffle on time-series destroys the
autocorrelation structure that all time-series models exploit; a PASS
under hard-shuffle is essentially "this is not a constant model", not
"this model uses real target alignment". Block-shuffle preserves
within-fold autocorrelation and forces the test to be about the
**predictive direction** of the alignment.

**Expected outcome.** Same chance baselines per fold; the verdict table
is per-fold.

## Operational pattern

```python
import numpy as np


def shuffle_test(
    fit_fn,                    # callable: (X_train, y_train) -> fitted model
    score_fn,                  # callable: (model, X_val, y_val) -> scalar
    X_train, y_train,
    X_val,   y_val,
    *,
    mode: str = "hard",        # "hard" | "within_group" | "block"
    groups=None,               # required for within_group / block
    n_repeats: int = 3,        # average over permutation seeds for stability
    seed: int = 0,
    chance_baseline: float | None = None,  # None → auto-derive
    tolerance: float = 0.05,   # PASS if |shuffled - chance| <= tolerance
):
    """
    Permute y_train, refit, evaluate on (X_val, y_val).
    Returns a verdict dict with PASS / WEAK / FAIL.
    """
    if chance_baseline is None:
        chance_baseline = _auto_chance_baseline(y_val, score_fn)

    real_score = score_fn(fit_fn(X_train, y_train), X_val, y_val)

    shuffled_scores = []
    rng = np.random.RandomState(seed)
    for _ in range(n_repeats):
        if mode == "hard":
            y_shuf = rng.permutation(y_train)
        elif mode == "within_group":
            y_shuf = _shuffle_within_groups(y_train, groups, rng)
        elif mode == "block":
            y_shuf = _shuffle_within_blocks(y_train, groups, rng)
        else:
            raise ValueError(f"Unknown shuffle mode: {mode!r}")
        shuffled_scores.append(score_fn(fit_fn(X_train, y_shuf), X_val, y_val))

    shuffled_mean = float(np.mean(shuffled_scores))
    shuffled_std  = float(np.std(shuffled_scores))
    distance      = abs(shuffled_mean - chance_baseline)

    verdict = (
        "PASS" if distance <= tolerance
        else "WEAK" if distance <= 5 * tolerance
        else "FAIL"
    )
    return {
        "verdict": verdict,
        "real_score": real_score,
        "shuffled_mean": shuffled_mean,
        "shuffled_std": shuffled_std,
        "chance_baseline": chance_baseline,
        "distance_to_chance": distance,
        "tolerance": tolerance,
        "n_repeats": n_repeats,
        "mode": mode,
    }
```

**Auto chance baseline derivation rules:**
- Binary classification, score = AUROC → `0.5`.
- Binary classification, score = accuracy → `max(p, 1-p)` where
  `p = mean(y_val)`.
- Multi-class classification, score = accuracy → `max(class_priors)`.
- Regression, score = R² → `0.0`.
- Ranking / trading metric → `0.0`.

## Where the result lands

The shuffle-test report is a sidecar file inside the winner archive:

- `winners/<tag>/shuffle_test.json` — machine-readable per-mode
  results (verdict, real_score, shuffled_mean, chance_baseline, etc.).
- `winners/<tag>/shuffle_test.md` — human-readable summary with the
  PASS / WEAK / FAIL banner and one line per mode.

Any project-level explainability or audit report that summarises the
winner's data-pipeline integrity MUST cite the shuffle-test verdict
alongside the structural split-audit fingerprint. A green structural
audit + FAIL shuffle test means the champion is NOT ready for an
external claim.

## Verdict tiers (binding)

| Verdict | Criterion | Required action |
|---|---|---|
| PASS | Shuffled metric within tolerance of chance baseline on all required modes | External claim is permitted (still subject to the statistical-rigor floor) |
| WEAK | 2×–5× tolerance distance from chance; not indistinguishable from chance but not at chance | Investigate the highest-importance feature for a partial label echo before claiming |
| FAIL | Shuffled metric > 5× tolerance OR within 2 std-devs of the real (unshuffled) score | STOP. No external claim. Fix the leakage, re-audit |

For grouped or temporal datasets, the Mode B or Mode C verdict
**overrides** the Mode A verdict — hard-shuffle on grouped or temporal
data is unreliable as a leakage detector.

## Anti-patterns

- **Shuffling only the validation targets** instead of the training
  targets. Shuffling `y_val` catches a different bug class (evaluator
  pathology) but does NOT catch the canonical leakage bug. Shuffle
  `y_train`, refit, evaluate on the **unshuffled** validation set.
- **Single-permutation verdict.** One permutation has high variance.
  Use `n_repeats >= 3` and average. A single shuffled metric near the
  chance baseline may be misleading; a 3-repeat mean with a reported
  standard deviation is the audit artefact.
- **Wrong chance baseline.** On imbalanced classification, the chance
  AUROC is still 0.5, but the chance accuracy is `max(p, 1-p)`, not
  0.5. Auto-derive from the validation distribution.
- **Shuffling features instead of targets.** Permuting rows of
  `X_train` destroys the row-feature pairing but preserves the global
  joint distribution — that is a different sanity check (feature
  dependence), not a leakage test.
- **Skipping the shuffle test because the structural split audit is
  green.** The structural audit is necessary but NOT sufficient. The
  canonical alignment bug had a green structural audit for weeks before
  the shuffle test exposed it.
- **Hard-shuffle on grouped data without Mode B.** False PASS is the
  failure mode — hard-shuffle destroys group-class correlations and
  makes the model appear to lose signal when it actually lost the group
  prior. Use Mode B for grouped CV.
- **Hard-shuffle on time-series without Mode C.** Same false-PASS
  mechanism — autocorrelation is what time-series models exploit;
  hard-shuffle destroys it.
- **Stopping at PASS without recording the shuffled metric value.** The
  shuffled metric is the audit artefact, not just the verdict. Future
  sessions re-run with the same seed and expect the same value — if it
  drifts, the loader changed.
- **Treating a shuffle-test FAIL as a hyperparameter to tune around.**
  A FAIL means the alignment is wrong. Tuning the model on a leaky
  pipeline is fitting to a corrupted signal.

## Cross-references

- [`autoresearch-data-split-audit`](../autoresearch-data-split-audit/SKILL.md)
  — structural leakage audit; the shuffle test is the **semantic**
  complement. Both must be green before an external claim.
- [`autoresearch-paper-rigor`](../autoresearch-paper-rigor/SKILL.md)
  — statistical-rigor floor for external claims; shuffle-test PASS is
  one of the four required legs of the dual-track audit.
- [`autoresearch-winner-archive`](../autoresearch-winner-archive/SKILL.md)
  — `shuffle_test.{json,md}` lives inside the archive alongside the
  structural audit report.
- [`autoresearch-experiment`](../autoresearch-experiment/SKILL.md)
  — the single-experiment ritual that produces the pipeline under
  audit; the shuffle test runs after Step 5 (Execute) but before any
  external claim is drafted.
- [`autoresearch-tiered-ladder`](../autoresearch-tiered-ladder/SKILL.md)
  — the shuffle test is required at the STANDARD rung and above; at
  SMOKE / DEV it is recommended but not blocking.

## Provenance

The shuffle-test pattern was elevated to a mandatory gate after a
retrospective on a case where a method showed a large, apparently
real uplift on the primary evaluation metric. The structural data-split
audit was green. The shuffle test was added retrospectively and FAILED
immediately — the shuffled metric was nearly as strong as the real
metric. Root cause: an off-by-one alignment between the training target
series and the evaluator's target series. After the fix, the real
uplift collapsed to near zero, and the downstream experiments built on
that result had to be discarded. The shuffle test is the lesson encoded
as a gate.
