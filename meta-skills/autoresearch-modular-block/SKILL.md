---
name: autoresearch-modular-block
description: Use when designing a composable module, pipeline stage, or configurable component that combines multiple independent design choices. Each design choice maps to a single Boolean flag that can be flipped in isolation. The mandatory smoke-test contract asserts every flag combination produces a valid output — if any combination fails, the block is not truly modular. Without clean flags, leave-one-out ablation is undefined.
---

# Skill — Ablation-friendly modular block with Boolean flags

## When to use

- When proposing a "block" or "module" that combines multiple
  independent ideas into one configurable component.
- When planning an ablation sweep: the sweep matrix writes itself
  once the flags are clean (single-flag-on + leave-one-out rows are
  automatic).
- When adding a new design choice to an existing block — the new
  choice is a new Boolean flag, not a hard-coded change.
- When a reviewer asks "which component contributed the headline
  number?" — a modular block answers this with a row from the sweep
  matrix; a monolithic block cannot.
- When implementing any component that is cited in a reasoning entry
  under the test discipline (see the meta-skills README): the flag
  combination smoke test is the UNIT-rung test.

## Why Boolean flags (not knobs)

A **Boolean flag** maps to a single binary design choice: component X
is included or excluded. A **knob** is a continuous or multi-valued
hyperparameter (learning rate, dropout rate, kernel size).

The distinction matters for ablations:
- Flags enable clean leave-one-out attribution: `flag_A = False`
  removes exactly one design choice.
- Knobs are hyperparameters; they belong in the project's tuning cube
  (see [`../autoresearch-per-hypothesis-hillclimb/SKILL.md`](../autoresearch-per-hypothesis-hillclimb/SKILL.md)),
  not in the ablation matrix.

A design choice that "only makes sense when another flag is on" is not
one flag — it is one design choice (the pair). Merge them into a single
flag.

## The flags contract

```python
# Illustrative pattern — adapt to the project's type system.

@dataclass
class ComponentFlags:
    """One Boolean per orthogonal design choice.
    
    Adding a flag is cheap; removing one mid-campaign breaks ablation
    comparability. Plan the flag set before the first sweep run.
    
    All flags default to False (the vanilla / literature-baseline state).
    """
    choice_a: bool = False   # e.g., a specific normalisation scheme
    choice_b: bool = False   # e.g., a specific weight-init strategy
    choice_c: bool = False   # e.g., a specific regulariser type
    # ... one per orthogonal design choice

    def tag(self) -> str:
        """Deterministic string tag for directory naming."""
        active = [k for k, v in vars(self).items() if v]
        return "+".join(active) if active else "vanilla"
```

Hard rules on the flags contract:

1. **All flags default to `False`.** The all-False state is the
   project's literature-anchored vanilla baseline — functionally
   identical to the reference architecture or pipeline.
2. **One flag = one orthogonal design choice.** If removing flag A
   requires also removing flag B to make sense, they are one choice.
   Merge them.
3. **Flags are not knobs.** `choice_a: bool` is a flag.
   `dropout_rate: float` is a knob. Do not label knobs as flags.
4. **Flag set is frozen before the first sweep.** Adding a flag
   mid-campaign invalidates comparisons with earlier single-flag-on
   rows.
5. **No silent compounding.** A flag's behaviour must not depend on
   another flag via a shared mutable. Surface any interaction as an
   explicit combined flag.

## The block structure

```python
class ComposableBlock:
    def __init__(self, ..., flags: ComponentFlags | None = None):
        flags = flags or ComponentFlags()
        self.flags = flags

        # Pre-dispatch sub-components from flags at construction time.
        # This keeps the forward/inference path static (no runtime branching).
        if flags.choice_a:
            self._component_a = ChoiceAImpl(...)
        else:
            self._component_a = DefaultAImpl(...)   # vanilla / identity

        if flags.choice_b:
            self._component_b = ChoiceBImpl(...)
        else:
            self._component_b = DefaultBImpl(...)
        # ...

    def forward(self, x):   # (or __call__, or process, or run — adapt to context)
        y = self._component_a(x)
        y = self._component_b(y)
        # ...
        return y
```

Pre-dispatch at construction time (not at forward time) keeps the
execution path static and makes the block easier to inspect,
serialise, and unit-test in isolation.

## Mandatory smoke test: every flag combination

```python
# Adapt to the project's testing framework and component signature.

import itertools

def test_all_flag_combinations():
    """
    If this test fails on any combination, the block is not truly modular:
    either the combination is not supported (document and exclude it) or
    there is a hidden dependency between flags (surface it as a merged flag).
    """
    flag_fields = [f.name for f in dataclasses.fields(ComponentFlags)]
    n_flags = len(flag_fields)

    for combo in itertools.product([False, True], repeat=n_flags):
        flags = ComponentFlags(**dict(zip(flag_fields, combo)))
        block = ComposableBlock(flags=flags)
        output = block.forward(make_test_input())

        assert output_is_valid(output), (
            f"Flag combo {flags.tag()!r} produced invalid output: {output!r}"
        )
```

This test is the UNIT-rung gate (Rung 0 in
[`../autoresearch-tiered-ladder/SKILL.md`](../autoresearch-tiered-ladder/SKILL.md)).
All flag combinations must pass before any background compute launches.
A runner that launches a sweep before this test is green has skipped
the plumbing check.

The test also validates the test discipline stated in the meta-skills
README: "every Boolean-flag combination" is exercised.

## The ablation matrix writes itself

Once the flags are clean, the sweep driver (see
[`../autoresearch-ablation-sweep/SKILL.md`](../autoresearch-ablation-sweep/SKILL.md))
produces the correct matrix automatically:

| row | flags | purpose |
|---|---|---|
| `baseline_reference` | reference architecture (no flags) | literature anchor |
| `baseline_vanilla` | all flags `False` | your pipeline zeroed |
| `only_choice_a` | `{choice_a: True}`, rest `False` | single-prior-on |
| `only_choice_b` | `{choice_b: True}`, rest `False` | single-prior-on |
| ... | ... | ... |
| `loo_no_choice_a` | all `True`, `{choice_a: False}` | leave-one-out |
| ... | ... | ... |
| `full` | all flags `True` | total effect |

The total row count is `2 + 2N + 1 = 2N + 3` for `N` flags. Pair with
3 seeds for the screening-level re-sweep of the best rows.

## Anti-patterns

| Anti-pattern | Why it fails |
|---|---|
| Flags whose default is not the vanilla state | The all-False state must be the lit-anchored baseline; any other default obscures the baseline comparison |
| Flags that silently multiply cost (e.g., double the parameter count) | Document the cost in the block's docstring AND in the ablation README; cost changes must be visible |
| Non-Boolean knobs labeled as flags | Knobs belong in the tuning cube, not the flag set; conflating them produces an uninterpretable matrix |
| Flags added mid-campaign | Breaks comparability with earlier single-flag-on rows; plan the flag set first |
| Runtime branching on flags inside the forward path | Pre-dispatch at construction time; dynamic branching makes serialisation and unit-testing harder |
| Skipping the all-combinations test | Any combination that silently fails is a hidden bug that will surface as a mysterious sweep crash |

## Cross-references

- [`../autoresearch-ablation-sweep/SKILL.md`](../autoresearch-ablation-sweep/SKILL.md)
  — the sweep skill that consumes a modular block's flag set to build
  the single-prior-on + leave-one-out matrix automatically.
- [`../autoresearch-tiered-ladder/SKILL.md`](../autoresearch-tiered-ladder/SKILL.md)
  — Rung 0 (UNIT) is the all-flag-combinations smoke test; the block
  must pass it before any sweep row runs.
- [`../autoresearch-experiment/SKILL.md`](../autoresearch-experiment/SKILL.md)
  — each row in the ablation matrix is one experiment under the 7-step
  ritual. The flag tag becomes the experiment tag.
- [`../autoresearch-data-contract-validator/SKILL.md`](../autoresearch-data-contract-validator/SKILL.md)
  — if the block flags affect the data pipeline (e.g., a flag changes
  the input preprocessing), re-run the data contract validator after
  any flag combination change that touches loaders or transforms.
- [`../autoresearch-combo-ladder/SKILL.md`](../autoresearch-combo-ladder/SKILL.md)
  — after ablation identifies positive single-flag-on results, the
  combo ladder tests whether multiple positive flags stack.
