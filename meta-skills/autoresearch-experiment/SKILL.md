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
   listed in the project's Dashboard Files Update Mandate.

## Hard rules

- **No `--bypass` flag.** If a gate refuses, fix the failing field.
- **Composite formula is SHA-256 fingerprinted.** Editing it breaks
  the project's comparability guarantee.
- **One config change per experiment.** Do not compound changes.
- **The experiment log is append-only.** Never rewrite a committed row.

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

## Cross-references

- [`autoresearch-reasoning-entry`](../autoresearch-reasoning-entry/SKILL.md)
  — Steps 1–4 and 6–7 are validated by this skill's word-count and
  citation-format gates.
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
