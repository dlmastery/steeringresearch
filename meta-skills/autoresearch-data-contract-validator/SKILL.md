---
name: autoresearch-data-contract-validator
description: Use before any training run to assert that the training pipeline's input-target pairing matches the evaluator's input-target pairing in shape, dtype, label encoding, and value range. This is the structural pre-run gate that catches the off-by-one alignment, the inverted-dimension-order, and the label-encoding divergence bugs — the class the semantic shuffle-test does NOT catch because it operates at the feature/label level, not the loader-contract level. The runner refuses to launch if the contract fails. There is no bypass flag.
---

# Skill — Pre-run (input, target) pairing data-contract validator

## When to use

- **Before any training run** where the training-side data loader and
  the evaluation-side data loader live in different functions, files,
  or processes. This is the single most common bug surface in
  iteratively-developed research pipelines: a refactor moves the eval
  loader to a new module and silently drops a transform, or uses a
  different indexing convention.
- **Before any deployment, paper submission, or external claim.** The
  contract validator runs as part of the runner's pre-flight, in the
  same layer as the structural split audit (see
  [`../autoresearch-data-split-audit/SKILL.md`](../autoresearch-data-split-audit/SKILL.md)).
- **After any change to the data loader, transforms, augmentation
  pipeline, or evaluator code path.** A change that looks local often
  silently shifts the input-target correspondence.
- **Before any smoke-test or UNIT rung run.** If the contract fails,
  the smoke test is meaningless — training and evaluation are not
  operating on a shared convention.

## Why the shuffle-test does NOT replace this

Three audits address different bug classes in the data pipeline:

| audit | what it catches | when it runs |
|---|---|---|
| structural split audit | Row-level train/val/test overlap | pre-run gate |
| **this contract validator** | Shape/dtype/encoding/alignment mismatch between loaders | pre-run gate |
| shuffle test | Semantic leakage (method works even on shuffled targets) | post-run gate |

The contract validator is **structural and static**: it catches the
bug class where the model trains fine and the evaluator runs fine but
the number reported is wrong because the two sides of the pipeline
are using incompatible conventions. Three concrete variants:

1. **Dimension-order divergence.** Training provides inputs in one
   channel ordering; evaluation silently provides them in another.
   The model trains on one convention and is evaluated on noise.
2. **Off-by-one alignment.** Training pairs `(input[i], target[i])`;
   the evaluator pairs `(input[i], target[i+1])`. On any dataset with
   autocorrelated targets, this produces a spurious evaluation signal.
3. **Label-encoding divergence.** Training uses one label encoding
   scheme; evaluation uses another (e.g., zero-indexed vs one-indexed,
   or opposite sign convention). The metric library interprets the
   mismatch silently and the reported accuracy or loss is garbage.

## The contract (what is asserted)

For every (training loader, evaluation loader) pair, the validator
asserts:

| property | assertion | bug class caught |
|---|---|---|
| Feature shape (excluding batch dim) | `eval_feat.shape[1:] == train_feat.shape[1:]` | Dimension-order mismatch, downsample mismatch |
| Feature dtype | `eval_feat.dtype == train_feat.dtype` | Silent cast divergence |
| Target shape | `eval_tgt.shape[1:] == train_tgt.shape[1:]` | Scalar vs multi-class encoding mismatch |
| Target dtype | `eval_tgt.dtype == train_tgt.dtype` | Integer width or sign overflow |
| Target-set membership | `set(unique(eval_tgt_sample)) ⊆ set(unique(train_tgt_sample))` | Label-encoding shift, unseen class in eval |
| Feature value range | `min(eval_feat) ≥ p1(train_feat) - δ` AND `max(eval_feat) ≤ p99(train_feat) + δ` | Silent normalisation difference |
| Pair-count match | `len(inputs) == len(targets)` in each loader | Off-by-one alignment (the most common variant) |
| Index-pair invariant (when applicable) | `loader[i]` returns the same `(input, target)` that iterating the loader yields at position `i` | Iterator vs indexer divergence |

### Modality-specific contracts (extend as needed)

| modality | additional contract |
|---|---|
| Image | Height and width match between train and eval loaders; channel ordering (CHW vs HWC) is consistent |
| Tabular | Column-name list matches; column dtypes match; no silent column drop or reorder |
| Time-series | Sequence length matches, OR the padding scheme is documented and the contract names it |
| Text | Tokenizer vocabulary and special-token IDs match between train and eval; max-length is consistent |
| Graph | Node-feature and edge-feature dimensions match; directionality convention matches |

### The index-pair invariant — the highest-leverage assertion

For any loader that exposes both a deterministic indexer (`__getitem__`)
and an iterator (`__iter__`), the contract asserts that iterating to
position `i` and indexing with `[i]` return the same pair. This is the
assertion that catches off-by-one alignment bugs early: if the iterator
and the indexer use different slicing windows, the mismatch is
detectable before a single training step.

Sample `k=8` random positions, compare `__iter__[i]` with `__getitem__(i)`.
The cost is negligible; the benefit is the single highest-leverage
assertion in the autoresearch pre-flight stack.

## Runner gate

The validator runs in the runner's pre-flight, immediately after the
structural split audit:

```
runner_preflight(cfg):
  1. structural_split_audit_or_die(cfg)     ← catches row-level overlap
  2. validate_data_contract(cfg)            ← this skill's gate
  3. smoke_test_or_die(cfg)                 ← catches plumbing
  4. ... onward to the 7-step ritual
```

On failure, the validator raises a hard exit (not a catchable
exception) that names the **specific contract violated** and the
**specific counterexample** (batch index, observed value, expected
value). This exit message must be actionable: a future session can
fix the loader without re-running the contract.

There is no `--skip-contract` flag, no environment variable that
disables the check. The contract is a hard gate. Smoke runs and
development runs are exactly when the contract matters most — the
contract catches bugs before they bake into an experiment log.

## Output artefacts

On pass, the validator writes `audits/data_contract.json` with:
- Status (`PASS`), timestamp, number of batches sampled, modality.
- Feature shape, feature dtype, target dtype confirmed.
- Target-set sizes for train and eval.
- Feature value range comparison.
- Number of index-pair invariant checks performed.
- Violation list (empty on pass).

On fail, the JSON includes the violation list verbatim. A companion
`data_contract.md` is the human-readable summary; commit it alongside
the structural split-audit report so a reviewer can confirm the contract
was honoured at the time of the experiment.

## Anti-patterns

| Anti-pattern | Why it fails |
|---|---|
| Checking only the first batch | Some augmentation pipelines produce contract-honouring first batches but contract-breaking later batches |
| Using `assert` instead of a hard exit | Test frameworks swallow assertions; `-O` strips them; the validator must crash the runner loudly |
| Skipping the index-pair invariant | This is the assertion that catches off-by-one alignment; its cost is negligible |
| Treating intended model-internal differences (e.g., running statistics) as contract violations | The contract is about input shape/dtype/range, not model-internal state |
| Adding a bypass flag for "fast runs" | Screening runs are exactly when the contract matters; there is no bypass |
| One validator per modality with hard-coded assumptions | The validator should be modality-aware (a parameter) not modality-specific; reuse across modalities |
| "I'll add the contract after I'm sure the pipeline works" | The pipeline only "works" if the contract holds; run from day one |

## Cross-references

- [`../autoresearch-data-split-audit/SKILL.md`](../autoresearch-data-split-audit/SKILL.md)
  — structural row-disjointness audit; runs before this contract
  validator. Both gate the runner. Recommended ordering: structural
  audit first (simpler bugs), then contract validator (harder bugs).
- [`../autoresearch-shuffle-test/SKILL.md`](../autoresearch-shuffle-test/SKILL.md)
  — semantic leakage detector; the contract validator catches
  alignment bugs **statically** (cheap pre-flight), the shuffle test
  catches semantic leakage **empirically** (expensive but more general).
  Both must pass before any external claim.
- [`../autoresearch-experiment/SKILL.md`](../autoresearch-experiment/SKILL.md)
  — the 7-step ritual's Execute step starts with the runner pre-flight;
  the contract validator is part of that pre-flight.
- [`../autoresearch-paper-rigor/SKILL.md`](../autoresearch-paper-rigor/SKILL.md)
  — a data-contract PASS is a gating criterion for the statistical-rigor
  floor to be meaningful.
- [`../autoresearch-modular-block/SKILL.md`](../autoresearch-modular-block/SKILL.md)
  — if the pipeline uses a modular block with Boolean flags, the
  contract validator should be re-run after any flag combination
  change that affects the data pipeline.
