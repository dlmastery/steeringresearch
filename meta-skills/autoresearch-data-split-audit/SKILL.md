---
name: autoresearch-data-split-audit
description: Use when wiring up a new dataset OR before launching ANY experiment on a dataset whose splits drive an external claim (OOD generalisation, benchmark-comparable test set, fixed-fold protocol). Runs a multi-auditor triple-check on the split — disjointness, protocol-conformance, class balance, size floors, reproducibility fingerprint, no-leakage-via-metadata — and writes a machine- and human-readable audit report. The project's runner refuses to launch if the audit is missing, stale, or failing. There is no --bypass-audit flag.
---

# Skill — Triple-check data-split audit gate

## When to use

- Wiring up a NEW dataset into the project (any modality: tabular,
  image, sequence, graph, or multi-modal).
- BEFORE the first experiment on any dataset whose splits drive an
  external claim — OOD generalisation, benchmark-comparable test
  performance, fixed-fold protocol.
- After ANY edit to the dataset loader, split configuration,
  transforms, or data-mode switch.
- On every session start that intends to run experiments — the audit
  is the canary that catches silent regressions in the data pipeline.
- After a crash recovery — pipeline state may be inconsistent.

## Why

A single leaked sample, group, or time-period can mask the entire
benchmark claim. For datasets whose purpose is cross-distribution
generalisation (e.g., leave-one-group-out protocols) or whose published
numbers all use a single frozen split (e.g., a standard last-N-rows
test set), a sloppy split is not a minor bug — it is a result that
cannot be reported.

This skill enforces the rule the project's history has validated:
**the project's runner refuses to launch without a green audit
fingerprint matching the live data loader.** There is no
`--bypass-audit` flag. Period.

## The auditor suite

Implement each auditor as a callable in the project's evaluation module
(e.g., `src/<pkg>/eval/audit.py` or `core/evaluation/audit.py`). Each
returns a status (`PASS` / `FAIL`), a violation list, and a SHA-256
fingerprint of its canonical artefact.

### Core disjointness auditors (every dataset)

1. **`audit_index_disjoint`** — every sample index appears in at most
   one fold. Pairwise intersection of all (train, val, test, ood_val,
   ood_test) sets must be empty:
   `set(train) ∩ set(val) == ∅`, and all symmetric pairings likewise.

2. **`audit_protocol_match`** — the canonical published or pre-agreed
   protocol for this dataset is honoured within tolerance. Examples
   of what this auditor checks (adapt to your project's datasets):
   - A standard frozen test set: the last N rows or a specific
     reserved partition must be used verbatim. Sizes must match
     within ±0.5%.
   - A leave-one-group-out protocol: each group must appear in
     exactly the folds the protocol specifies; no group may appear
     in two folds simultaneously.
   - A random-split reproducibility requirement: the same seed and
     split ratio must produce the same split on every run.

3. **`audit_size_floors`** — every fold meets a minimum sample count,
   configured per dataset in the split config. Violation example:
   "val fold has 100 samples; floor is 10,000."

### Sanity auditors

4. **`audit_class_balance`** — every classification fold has all
   classes present (aggregate metrics such as AUROC are otherwise
   undefined) AND positive prevalence is within the dataset's known
   range. Document the expected range in the split config.

5. **`audit_no_leakage_via_metadata`** — confirms that columns used
   for split assignment (group ID, entity ID, date, row index) are
   NOT among the model's input features. No split-key information
   should be visible to the model at inference time.

6. **`audit_feature_consistency`** — same feature names and data types
   across all splits; no NaN or Inf in any feature column; any
   normalisation (mean-centering, scaling) uses **training-set
   statistics only**. Computing mean or variance from the validation
   or test partition is data leakage.

### Modality-specific extras

For datasets with hierarchical grouping (e.g., multi-slide imaging,
multi-hospital studies, multi-subject experiments):

7a. **`audit_group_level`** — every group identifier appears in at
   most one fold; pairwise group intersections are empty. Samples
   drawn from the same group cannot appear in different folds.

7b. **`audit_subgroup_level`** — every subgroup (institution, source
   domain, recording day) appears in exactly the folds the protocol
   assigns to it.

For time-series and sequential data:

7c. **`audit_temporal_order`** — train end-time < val start-time and
   val end-time < test start-time, with any required purge gap and
   embargo enforced. Document the gap requirements in the split config
   (e.g., purge gap, label-horizon buffer).

7d. **`audit_no_lookahead`** — no feature in sample[T] uses
   information from any time > T. Verified by re-deriving each feature
   using only the lagged window and checking equality with the stored
   feature.

For graph-structured data:

7e. **`audit_node_edge_disjoint`** — depending on the split mode
   (transductive vs. inductive), confirm either edge-set or node-set
   disjointness across folds, with the chosen mode logged explicitly.

### Reproducibility auditor (every dataset)

8. **`audit_reproducibility`** — running the audit twice with the same
   seed must produce identical `(sizes, sample_set_fingerprints,
   feature_names)` tuples, captured as a SHA-256 hash. Any change
   between runs is a regression — most likely a non-deterministic
   loader or a stochastic preprocessing step.

## Output artefacts

The audit writes to a project-relative path (recommended:
`<results_root>/audits/data_split_audit/`):

| file | content | format |
|---|---|---|
| `data_split_audit.json` | machine-readable per-auditor results | JSON |
| `data_split_audit.md` | human-readable summary + violation list | Markdown |
| `data_split_audit_fingerprint.json` | SHA-256 of the canonical split | JSON |

The `.md` report is checked into version control so a reviewer can
read it without re-running the audit.

## Runner gate (the load-bearing rule)

The project's training runner MUST call `audit_or_die()` before any
model build or expensive computation. Pseudocode:

```python
def audit_or_die(cfg, results_root) -> None:
    audit_path = results_root / "audits" / "data_split_audit.json"
    if not audit_path.exists():
        raise SystemExit(
            "REFUSED: data-split audit missing. "
            "Run: python -m <pkg>.eval.audit --config <cfg>"
        )
    audit = json.loads(audit_path.read_text())
    if audit["age_hours"] > 24:
        raise SystemExit("REFUSED: data-split audit is stale (> 24 h).")
    failed = [a for a in audit["auditors"] if a["status"] != "PASS"]
    if failed:
        raise SystemExit(
            f"REFUSED: auditor(s) failed: "
            f"{[a['name'] for a in failed]}. "
            f"Read {results_root}/audits/data_split_audit.md."
        )
    live_fp = compute_split_fingerprint(load_data(cfg))
    if live_fp != audit["fingerprint"]:
        raise SystemExit(
            f"REFUSED: live data fingerprint {live_fp} != "
            f"audited fingerprint {audit['fingerprint']}. Re-run audit."
        )
```

`SystemExit` (not a return code) so calling shell scripts surface the
violation. There is no `--bypass-audit` flag — fixing the violation is
the only path forward.

## When to re-run the audit

| event | re-run? |
|---|---|
| Session start, any experiment planned | YES — cheap; catches drift |
| Edit to dataset loader, transforms, or split config | YES |
| Switch between data modes (e.g., subset ↔ full, simulation ↔ real) | YES |
| Upgrade of any data-loading library version | YES |
| Crash mid-experiment | YES — pipeline state may be inconsistent |
| Second run within an unchanged session on the same data | NO — fingerprint will match |

## Audit failure protocol

When an auditor reports `FAIL`:

1. **Read the violation list** in `data_split_audit.md` — it names
   the specific assertion and the counter-example (e.g., "sample ID
   12345 appears in both train and val").
2. **Do NOT silence the audit.** The audit is the canary, not the
   bug. Suppressing the auditor with a try/except is a `--bypass` by
   another name.
3. **Fix the underlying** loader, split config, metadata handling, or
   standardisation mismatch.
4. **Re-run the audit** until all auditors are green.
5. **Commit the fix and the green audit report together** so the green
   row in version history is traceable to the patch.

## What "good" looks like

- The `data_split_audit.md` report opens with a single PASS/FAIL
  banner followed by a table of (auditor name, status, fingerprint
  snippet, violation count).
- The reproducibility fingerprint is short (e.g., 12 hex characters)
  and stable across consecutive re-runs.
- The runner's refusal message names the specific failed auditor.
- A pre-commit hook (optional but recommended) refuses to commit
  changes to the dataset loader without a fresh green audit report.

## Anti-patterns

- **Silencing an auditor with `# noqa`, a try/except, or a config
  flag.** That is a `--bypass` by another name.
- **Lowering a size floor to make a FAIL pass.** Either the dataset
  shrank legitimately (update the floor in config and document why) or
  the loader broke.
- **"I'll add the audit later" for exploratory experiments.** The
  exploratory runs are the ones most likely to bake a split bug into
  later analysis. Run the audit from day one; it is cheap.
- **Computing normalisation statistics from the validation or test
  split.** That is data leakage, regardless of whether the structural
  split is otherwise clean.
- **Treating a public leaderboard test set as a held-out evaluator**
  if hyperparameters were tuned against scores obtained from that
  test set. It is not held-out if it was used for tuning.

## Cross-references

- [`autoresearch-experiment`](../autoresearch-experiment/SKILL.md)
  — the runner's `audit_or_die()` precedes Step 5 (Execute) of the
  7-step ritual. No experiment runs without a green audit.
- [`autoresearch-shuffle-test`](../autoresearch-shuffle-test/SKILL.md)
  — the semantic complement to this structural audit. Structural +
  semantic must both be green before an external claim.
- [`autoresearch-paper-rigor`](../autoresearch-paper-rigor/SKILL.md)
  — the statistical-rigor floor; a green split audit is a
  prerequisite for the rigor floor to be meaningful.
- [`autoresearch-tiered-ladder`](../autoresearch-tiered-ladder/SKILL.md)
  — the split audit gate applies at every rung, not just the final
  evaluation rung.
- [`autoresearch-experiment`](../autoresearch-experiment/SKILL.md)
  — the pre-run reasoning entry (part of the 7-step ritual in the
  experiment skill) should cite the audit fingerprint as the provenance
  of the train/val/test split.
