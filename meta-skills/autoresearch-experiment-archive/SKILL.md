---
name: autoresearch-experiment-archive
description: Use when archiving any experiment that produced a numeric result worth preserving. Defines the taxonomy directory structure and the mandatory detailed README that every archived experiment must carry so that a future session can reproduce the experiment cold from just that subdirectory. The per-experiment archive is the unit of reproducibility; the winner archive (autoresearch-winner-archive) is additive on top of it for global-champion experiments.
---

# Skill — Per-experiment archive taxonomy

## When to use

- After every experiment that produced a numeric result you want to
  preserve — which is almost every experiment in a principled
  autoresearch program.
- Before marking an experiment `KEEP`, `NEAR-MISS`, or `DISCARD` in the
  experiment log: the archive must exist before the verdict is written,
  because the verdict references the archive path.
- After a global-best result: the per-experiment archive is created
  first, then the winner archive (see
  [`../autoresearch-winner-archive/SKILL.md`](../autoresearch-winner-archive/SKILL.md))
  is created on top. The per-experiment archive stays in place; the
  winner archive is additive, not a replacement.
- When a future session asks to reproduce an old experiment: the archive
  is the unit of cold reproducibility. If the archive is incomplete,
  the experiment cannot be reproduced without re-running it.

## The unit of archive

One experiment = one subdirectory. The canonical location:

```
ideas/<NN_hypothesis_name>/experiments/expNNN_<short_name>/
```

Everything for ONE experiment lives below this path. No global
side-files, no symlinks into other hypotheses, no shared config files
that live outside the archive and could drift.

The experiment tag (`expNNN_<short_name>`) matches the tag in the
project's `experiment_log.jsonl`. The `NNN` is zero-padded and
monotone-increasing across the hypothesis's experiments.

## Mandatory contents

```
expNNN_<short>/
├── README.md              ← the very-detailed design doc (see below)
├── config.yaml            ← exact config passed to the runner (frozen)
├── reasoning.json         ← citation-gated 7-step reasoning entry
├── run_seed0/
│   ├── metrics.json       ← per-axis numeric results
│   ├── history.json       ← per-step training or evaluation log
│   └── checkpoint.<ext>   ← model or state checkpoint (if applicable)
├── run_seed1/             ← if multi-seed
├── run_seed2/
└── dashboard/
    ├── dashboard.html     ← self-contained per-experiment dashboard
    └── plot_*.png         ← relevant plots (Pareto, curves, etc.)
```

Every file is a real file in the directory. No symlinks. The archive
must be self-contained: someone with only this subdirectory must be
able to read the experiment and reproduce it.

## The mandatory detailed README

The README is the primary deliverable of the archive. It must contain
all of the following sections — even if a section is short, it must
exist:

```markdown
# expNNN — <one-line title>

## TL;DR

2–3 sentences: the headline metric, the verdict, and the key learning.

## 1. Motivation

Why this experiment exists. Which hypothesis (idea NN) it is testing.
Reference the parent hypothesis README for context.

## 2. Hypothesis

The mechanism being claimed. What changes and why the mechanism
predicts an improvement. Word count ≥ 50.

## 3. Citations

Full citation block (Author, Year, Venue, Title, Identifier, one-
sentence relevance note) for every paper motivating the change.
See the citation format in autoresearch-experiment.

## 4. Pre-registered prediction

A numeric range on the composite AND at least one sub-metric. This
section must have been written BEFORE the run started; it is the
pre-registration. Word count ≥ 25.

## 5. Method

- Exact configuration: link to config.yaml
- Dataset name + train/val/test sizes + rung
- Optimiser, schedule, batch size, precision, seeds
- Composite formula + SHA-256 fingerprint

## 6. Results

| seed | <primary_axis> | <axis_2> | ... | composite |
|------|----------------|----------|-----|-----------|
| 0    | ...            | ...      | ... | ...       |
| 1    | ...            | ...      | ... | ...       |
| 2    | ...            | ...      | ... | ...       |

Include inline plots as `![<caption>](dashboard/<plot>.png)`.

## 7. Verdict

KEEP / NEAR-MISS / DISCARD + reasoning. Word count ≥ 30.
Reference the pre-registered prediction and say whether it was met.

## 8. Learning

What the project now believes that it did not before this experiment.
What to try next. Word count ≥ 40.

## 9. How to reproduce

Exact commands, from a clean clone, to reproduce this experiment and
rebuild the dashboard. No hidden prerequisites.

## 10. Open issues

Anything that surprised the researcher, any caveat about this
experiment's results, any known limitation. Bullet list.
```

Sections 1–4 are pre-run content (written before the run). Sections 6–10
are post-run content. Section 5 bridges them (written before the run;
updated with actual fingerprint after the run). The runner gate
prevents launching if any pre-run section is a placeholder.

## Hard rules

1. **Archives are immutable after the verdict is written.** The archive
   is append-only from that point: if a result needs re-interpretation
   or correction, create `expNNN_<short>_v2/` with a link back to the
   original. Never edit a completed archive in place.
2. **No symlinks.** Every artifact is a real file in the directory.
   Symlinks break when the directory is moved or copied.
3. **The README is the source of truth.** If the local `dashboard.html`
   disagrees with the README's results table, the README wins until
   the two are reconciled.
4. **Each archive has its own local dashboard.** The `dashboard/` subdir
   is self-contained so the archive can be inspected without the global
   dashboard toolchain. The global dashboard also links to it.
5. **Total directory size must be manageable.** Large artifacts (model
   weights, large datasets) should be compressed or stored in a
   designated artifact store and linked by URI in the README, not
   stored as large binary files in the git repository.

## Relationship to the winner archive

The per-experiment archive is created for EVERY experiment. The winner
archive (see [`../autoresearch-winner-archive/SKILL.md`](../autoresearch-winner-archive/SKILL.md))
is created ONLY for experiments that are also the new global champion.
The relationship is additive:

```
experiments/expNNN_<short>/         ← per-experiment archive (always)
    └── (normal archive contents)

winner_archive/<champion_tag>/      ← winner archive (global-best only)
    ├── frozen_config.yaml          ← frozen at the moment of archival
    ├── champion_pointer.json       ← points back to experiments/expNNN_<short>/
    └── (additional audit report, inference script, etc.)
```

The per-experiment archive stays in place when the winner archive is
created. The winner archive is a separate directory that adds provenance
and frozen-config discipline on top.

## Anti-patterns

| Anti-pattern | Why it fails |
|---|---|
| Writing the README after the experiment completes | The pre-run sections (hypothesis, prediction) become post-hoc reconstructions — HARKing risk |
| Mixing two experiments in one archive subdirectory | Breaks the one-experiment-per-directory invariant; attribution is undefined |
| Omitting sections from the mandatory README | A future session cannot reproduce the experiment without all sections; an incomplete archive is dead weight |
| Using symlinks to shared configs | The archive is no longer self-contained; shared configs can drift |
| Storing large binary artifacts in git without compression or an artifact store | Repository size bloat; makes clone and fetch slow |
| Not creating a local `dashboard/dashboard.html` | The archive requires the global dashboard toolchain to inspect; not self-contained |

## Cross-references

- [`../autoresearch-experiment/SKILL.md`](../autoresearch-experiment/SKILL.md)
  — the 7-step ritual that generates the reasoning.json content and
  the verdict. The archive is the storage container for the ritual's
  outputs.
- [`../autoresearch-winner-archive/SKILL.md`](../autoresearch-winner-archive/SKILL.md)
  — the additive layer for global-champion experiments. Created on top
  of the per-experiment archive, not as a replacement.
- [`../autoresearch-per-experiment-page/SKILL.md`](../autoresearch-per-experiment-page/SKILL.md)
  — the dashboard page generated from this archive's metrics and
  reasoning blob. The archive is the source of truth for the page.
- [`../autoresearch-idea-scaffold/SKILL.md`](../autoresearch-idea-scaffold/SKILL.md)
  — the per-hypothesis subdirectory scaffold that includes an
  `experiments/` directory; each archive is created within that
  scaffold.
- [`../autoresearch-doc-organization/SKILL.md`](../autoresearch-doc-organization/SKILL.md)
  — the repo-root discipline that specifies `experiments/` (or
  `ideas/<NN>/experiments/`) as the correct home for archive
  subdirectories.
- [`../autoresearch-checkpoint/SKILL.md`](../autoresearch-checkpoint/SKILL.md)
  — commit the archive directory as part of the experiment's checkpoint.
  The per-experiment commit commits the archive directory, not scattered
  files.
