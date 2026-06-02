---
name: autoresearch-experiment
description: Use when running ONE principled experiment. Enforces the 7-step ritual (diagnose, cite, hypothesise, predict, execute, analyse, checkpoint) with Citation Rigor and Reasoning Blob Completeness gates and a Goodhart-fingerprinted composite metric.
---

# Skill — Run one autoresearch experiment

## When to use

Any time a single experiment is going to produce a numeric result that
should later survive third-party scrutiny. Single config change, single
hypothesis, single seed-set.

## The 7-step ritual

Read AUTORESEARCH_PROCESS.md if it exists; otherwise these are the
fields:

1. **Diagnose** (>= 60 words). Read the last completed row from
   the project's experiment log (typically `experiment_log.jsonl`).
   Identify the specific failure mode or open question. Reference at
   least one prior experiment by tag OR per-row metric value.

2. **Cite** (>= 40 words for a single paper / >= 80 words for
   multiple papers). Search the relevant literature venues for the
   paper that motivates the change. Format every citation as:
   ```
   Author1, Author2, ..., YEAR VENUE 'Title'
   (arXiv:XXXX.XXXXX or DOI) -- one-sentence relevance note.
   ```

3. **Hypothesise** (>= 50 words). State the mechanism: what parameter
   moves, what it does in the model or pipeline, what the cited paper
   predicts. The hypothesis text must contain at least one of the words
   "mechanism", "because", or "per [paper]".

4. **Predict** (>= 25 words). Numeric outcome range on the composite
   metric plus at least one sub-metric prediction (e.g., per-fold
   performance, error rate on the held-out split, generalisation gap).
   Predictions are stored **before** training or computation begins.

5. **Execute.** ONE config change per experiment. The project's runner
   enforces this through the per-experiment-number reasoning entry — it
   will not launch the next experiment until this one's verdict is
   written.

6. **Analyse** (>= 30 words on verdict). Compare actual to predicted.
   Update the verdict field with `KEEP` / `DISCARD` / `NEAR-MISS`,
   the exact composite metric to 4 decimal places, the delta versus
   the global best, and per-fold or per-split narrative.

7. **Checkpoint** (>= 40 words on learning). Update every artifact
   listed in the project's Dashboard Files Update Mandate. Additionally:
   - Regenerate (or update) `PROVENANCE.md` in the relevant hypothesis
     sub-project (`ideas/<NN>/PROVENANCE.md`) to record this experiment's
     tag, command, result path, and a 2–3 sentence result interpretation.
   - If EXPERIMENT_LEDGER.md is updated with a new verdict, verify that
     the campaign-arc narrative in the ledger's "how to read" section
     is still accurate and update it if a phase boundary has been crossed.
   - If a finding graduates to FINDINGS.md, confirm the preamble glossary
     still defines every identifier used in the new finding.

## L6 — Mandatory controls in every effect-claim experiment

Every experiment that claims a directional or causal effect must beat at
least **two controls** beyond the do-nothing (null-intervention) baseline:

**Control A — Matched-magnitude random-direction baseline:**
Apply an intervention of the same magnitude as the experimental intervention,
but in a random direction (drawn fresh for each experiment). Report the
benefit delta over this control, not over the null baseline. A real
directional effect must exceed a random perturbation of the same magnitude.

*Why it matters:* any sufficiently large perturbation can change output
statistics in ways that inflate metrics measured by pattern-matching or
correlation proxies. If the random-direction control scores nearly as high
as the intended direction, the effect is not directional — it is a
magnitude artifact.

**Control B — Semantically-inverted / shuffled-label baseline:**
Apply the intervention using the OPPOSITE or semantically-inverted version
of the intervention signal (e.g., flipped labels, reversed contrast,
permuted condition). The benefit score on this inverted control must be
significantly lower than on the true intervention. If not, the metric is
not detecting a real directional effect.

*Why it matters:* this is the generalization of the shuffle test to within-
experiment controls. A metric that rewards both a direction and its near-
opposite is measuring something other than the intended effect.

**Reporting rule:** every experiment row in the project's ledger must include:
- `delta_vs_random_magnitude`: benefit − benefit of matched-magnitude random
- `delta_vs_inverted_signal`: benefit − benefit of inverted/shuffled signal

A claimed effect requires BOTH deltas to be positive. An experiment where
`delta_vs_random_magnitude ≤ 0` is DISCARD regardless of raw benefit.

## L10 — Extraction stability gate (before any direction-based downstream use)

When an experiment uses a direction or basis extracted from a limited sample
(contrast pairs, principal components, few-shot examples), that extracted
direction may be unstable: a different random draw of the same sample size
can yield a substantially different direction, making all downstream results
noise.

**Bootstrap-stability gate (domain-agnostic):**

1. Bootstrap the extraction sample K ≥ 100 times (sample with replacement).
2. For each bootstrap, re-extract the direction.
3. Compute pairwise cosines between each bootstrap direction and the original.
   Report `stability_p5` — the 5th-percentile cosine (the stability floor).
4. **Gate:** if `stability_p5 < 0.85`, the direction is UNSTABLE. Do NOT
   use it downstream until stability clears (more data or a more robust
   estimator).

Log `stability_p5` and sample size in the experiment ledger row for any
experiment that depends on an extracted direction.

**Near-tautological estimator caveat:** when two estimators are computed on
a very small sample and agree closely, this is often a tautology (the sample
is too small to discriminate the estimators), not a quality indicator. Tag
such rows as `NEAR_TAUTOLOGICAL` and do not cite their agreement as evidence
of correctness.

## Hard rules

- **No `--bypass` flag.** If a gate refuses, fix the failing field.
- **Composite formula is SHA-256 fingerprinted.** Editing it breaks
  the project's comparability guarantee.
- **One config change per experiment.** Do not compound changes.
- **The experiment log is append-only.** Never rewrite a committed row.
- **Two mandatory controls for every effect-claim experiment** (L6):
  matched-magnitude random + inverted/shuffled signal. An experiment
  without these controls cannot claim a directional effect.
- **Extraction stability gate** (L10): any experiment depending on an
  extracted direction must report stability_p5 ≥ 0.85. Do not proceed
  downstream with an unstable direction.

## How to execute (typical commands)

The exact module paths depend on the project. Adapt the following
skeleton to your codebase:

```bash
# 1. Author reasoning entry (pre-run)
$EDITOR ideas/<NN_idea>/experiments/expNNN_<short>/reasoning.json
# required fields: diagnosis, citations[], hypothesis, prediction

# 2. Validate it via the project's reasoning module
python -c "from <project>.reasoning import validate_entry, load_all; \
  e=load_all('.../reasoning.json')[-1]; \
  print(validate_entry(e))"
# Expect: []   (empty list = zero validation errors)

# 3. Run the experiment (the runner re-validates on launch)
python -m <project>.runner \
  --config ideas/<NN_idea>/experiments/expNNN_<short>/config.yaml \
  --tag expNNN_<short> --seed 0 \
  --root ideas/<NN_idea>/experiments/expNNN_<short>/run

# 4. Author verdict + learning (post-run)
$EDITOR ideas/<NN_idea>/experiments/expNNN_<short>/reasoning.json

# 5. Regenerate dashboards
python scripts/build_dashboard.py
python scripts/build_report.py
```

## What "good" looks like

- Pre-run reasoning entry validates with zero errors.
- The composite metric landed inside the predicted range (KEEP) or
  within 2 percentage points (NEAR-MISS).
- Verdict and learning sections are concrete enough that a future
  contributor can pick up the campaign cold without context.
- The Dashboard Files Update Mandate is fully satisfied — no orphan
  artifacts.

## When to skip this skill

- Pure infrastructure work that produces no numeric result (e.g.,
  refactoring the data loader). Add a `chore:` commit instead.
- Truly throwaway debug runs — place them under a dedicated debug
  directory (e.g., `experiments/debug/`) so they cannot pollute the
  main experiment log.

## Anti-patterns to refuse

- "Let me bypass the gate just this once."
- Adding a parameter-penalty term to the composite metric mid-project
  to artificially crown a favoured row.
- Cherry-picking the best of N seeds and reporting that as the
  headline result.
- Citing `(Author2024)` without title, venue, and arXiv/DOI.
- Running two simultaneous config changes and attributing the combined
  delta to one of them.
- Writing the prediction field after seeing the result.
- Claiming a directional effect without both mandatory controls (L6).
  "Beats null baseline" is not sufficient — beats random-direction AND
  beats inverted-signal.
- Using an extracted direction without checking stability_p5 (L10).
  An unstable direction makes the experiment's results uninterpretable.
- Citing high agreement between two estimators on a tiny sample as
  quality evidence (near-tautology caveat, L10).

## Cross-references

- (The reasoning-entry gates are part of this skill — Steps 1–4 and
  6–7 and their word-count and citation-format validation are defined
  in this SKILL.md directly, not in a separate skill.)
- [`autoresearch-findings-ledger`](../autoresearch-findings-ledger/SKILL.md)
  — the Checkpoint step (Step 7) must satisfy the provenance-tracing
  requirements and the ledger/findings self-contained mandates.
- [`autoresearch-idea-scaffold`](../autoresearch-idea-scaffold/SKILL.md)
  — the PROVENANCE.md file that must be updated at every Checkpoint.
- [`autoresearch-data-split-audit`](../autoresearch-data-split-audit/SKILL.md)
  — Step 5 (Execute) calls the runner's `audit_or_die()` before any
  model build. New dataset? Run the audit first.
- [`autoresearch-winner-archive`](../autoresearch-winner-archive/SKILL.md)
  — if the verdict is KEEP and the composite beats the global best,
  Step 7 (Checkpoint) triggers winner archiving.
- [`autoresearch-session-resume`](../autoresearch-session-resume/SKILL.md)
  — Step 7 also updates the crash-recovery checkpoint document so the
  next session can resume without context loss.
- [`autoresearch-per-hypothesis-hillclimb`](../autoresearch-per-hypothesis-hillclimb/SKILL.md)
  — after a screening sweep produces a candidate, the hill-climb is
  the proper evaluation tier before any external claim.
- [`autoresearch-paper-rigor`](../autoresearch-paper-rigor/SKILL.md)
  — the statistical-rigor floor that every external claim must satisfy.
- [`autoresearch-tiered-ladder`](../autoresearch-tiered-ladder/SKILL.md)
  — the rung at which this experiment sits determines the burden of
  proof; a UNIT or SMOKE result cannot carry an external claim.
