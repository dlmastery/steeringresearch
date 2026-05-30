# VERIFY — H<NN>

> Verification log. Must be signed before the first experiment is archived.
> Do NOT sign before all tests actually pass — signing a failing VERIFY.md
> is a protocol violation.

## Tests

- [ ] `tests.py` passes all assertions (run: `pytest ideas/<NN>_<name>/tests.py -v`)
- [ ] No new linter warnings on `implementation.py`
- [ ] No new type-check errors on `implementation.py`

## Sanity checks

- [ ] Vanilla / disabled flag combination produces identical output to the
      documented baseline (zero edit = zero effect)
- [ ] All flag combinations run forward without shape or type errors (smoke: 5 prompts)
- [ ] Resource cost (VRAM, wall-clock, parameter count) within +/-10% of predicted
- [ ] Activation cache is NOT recomputed inside the sweep loop (efficiency check)
- [ ] Extraction pairs are NOT in the eval set (data-split audit passes)
- [ ] Unit-normed vectors throughout; alpha is the only magnitude control

## Geometry sanity

- [ ] Off-shell displacement delta_norm is logged and within expected range at test alpha
- [ ] Effective-rank probe runs without error
- [ ] Norm-budget fraction (N5) is logged

## AUDIT.md sign-off

- [ ] All HIGH-severity items in AUDIT.md are resolved in IMPROVEMENTS.md
- [ ] IMPROVEMENTS.md fix log has no unchecked HIGH items

## Signed off by

Author: <identifier>
Date: <YYYY-MM-DD>
Git commit at sign-off: <sha>
