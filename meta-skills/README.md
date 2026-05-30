# meta-skills/

> The **portable, topic-agnostic autoresearch process**. This pack encodes a
> publication-grade research loop — the Karpathy keep/discard core, the 7-step
> ritual, the CIFAR-style benchmark ladder, the Goodhart-resistant composite, the
> screening→hill-climb→evaluation funnel, the dual-track audit, and the
> hierarchically-linked dashboard — so **any** future topic can pick it up
> unchanged. None of these skills mention a specific research domain. They
> describe the *protocol* and the *infrastructure contract*, never the content.

The centerpiece is [`autoresearch-meta/SKILL.md`](autoresearch-meta/SKILL.md):
the full spine that ties every supporting skill into one coherent program. Read
it cover-to-cover before instantiating the process on a new topic. Each
supporting skill is one workflow extracted from that spine.

---

## How to use

1. Read [`autoresearch-meta/SKILL.md`](autoresearch-meta/SKILL.md) — the spine.
   It contains the "instantiate on a new topic in 10 steps" checklist and the
   definition-of-done self-audit.
2. Define your topic's **axes**, **composite**, **datasets-per-rung**, and
   **runner contract** (spine §"Instantiate in 10 steps").
3. Pull in each supporting skill as the program grows: start with
   `autoresearch-idea-scaffold` + `autoresearch-experiment`, add the ladder,
   hill-climb, dashboard, audit, and combo skills in turn.

Skills are progressive. You do not need all of them on day one; you need the
spine plus `autoresearch-experiment` to run your first principled experiment.

---

## The meta-skill pack (index)

| skill | when to use |
|---|---|
| **[autoresearch-meta](autoresearch-meta/SKILL.md)** | The spine. Any time you start or run a rigorous autoresearch program on a new topic. Defines axes, composite, ladder, funnel, dashboard, audit, and the 10-step instantiation. |
| **autoresearch-experiment** | Running ONE principled experiment with the 7-step ritual (diagnose → cite → hypothesise → predict → execute → analyse → checkpoint) under the word-count + citation-format gates. |
| **autoresearch-tiered-ladder** | Wiring and enforcing the 5-rung UNIT/SMOKE/DEV/STANDARD/FULL benchmark ladder and its promotion gates ("same axes at every rung, only size/realism grow"). |
| **autoresearch-per-hypothesis-hillclimb** | After a single-config screening sweep surfaces a candidate, the proper evaluation tier: a 20–25-trial coordinate-descent hill-climb over the tuning cube + an n≥7-seed confirmation. |
| **autoresearch-paper-rigor** | Enforcing the statistical-rigor floor (paired Wilcoxon + bootstrap CI + Holm-Bonferroni + empirical noise band), pre-registration, no-HARKing, and the verdict tiers before any external claim. |
| **autoresearch-shuffle-test** | The label/condition shuffle (negative-control) check: a method must NOT "work" on shuffled labels. Run before any external claim. |
| **autoresearch-data-split-audit** | The `audit_or_die()` leakage / split-integrity check run before any model build on a new dataset (train/val/test disjointness, no target leakage, no duplicate rows across splits). |
| **autoresearch-dashboard** | Generating the self-contained, sortable master dashboard with radar / Pareto / ladder-board / stack-matrix panels and mandatory sub-links to sub-dashboards. |
| **autoresearch-dashboard-comprehension** | The "how to read this" orientation discipline: every dashboard ships a 4-bullet reader's guide and a tier/`n=` chip on every numeric cell. |
| **autoresearch-per-experiment-page** | Rendering the per-experiment leaf page (full 7-step reasoning entry, sweep curves, side-by-side samples, all axes with CIs) that the dashboard links down to. |
| **autoresearch-typography-and-rendering** | The rendering hard rules: markdown actually rendered (no literal `##`/`**`/`\|---\|` leak), PNG not SVG, self-contained HTML, no-emoji default. |
| **autoresearch-link-discipline** | HEAD-testing every absolute repository-blob link; no dead links; sub-dashboard links mandatory and verified. |
| **autoresearch-critic-team** | Dispatching the implementation-critic agents (does the code do what the reasoning entry claims?) with the same-model-family circularity disclosure. |
| **autoresearch-scicritic-team** | Dispatching the science-critic agents (is the *claim* sound — NUMEROLOGY? UNFALSIFIABLE? HARKed?) with the circularity disclosure. |
| **autoresearch-fixer-campaign** | Running a bounded Fixer campaign when a critic returns BROKEN: fix → re-screen → re-hill-climb to re-establish the tuned ceiling. |
| **autoresearch-multi-agent-dispatch** | Fanning out N agents with disjoint file scopes, scoped `git add <paths>`, retry-wrapped commits, and bounded structured returns. |
| **autoresearch-checkpoint** | The commit+push heartbeat: the milestone trigger table that guarantees a crash never loses progress. |
| **autoresearch-session-resume** | Authoring/consuming the crash-recovery checkpoint document so the next session resumes cold without context loss. |
| **autoresearch-idea-scaffold** | Scaffolding a new hypothesis sub-project from `ideas/_TEMPLATE/` (config, reasoning, experiments/, dashboard/). |
| **[autoresearch-combo-ladder](autoresearch-combo-ladder/SKILL.md)** | Building the orthogonal-axis additive 2→N stacking ladder (one new orthogonal prior per row; the "everything-on" hybrid is forbidden). |
| **[autoresearch-ablation-sweep](autoresearch-ablation-sweep/SKILL.md)** | The structured screening sweep (baselines + single-prior-on + leave-one-out + curated subset) that identifies which design choices are worth hill-climbing. The screening counterpart to the hill-climb evaluation tier. |
| **[autoresearch-auto-checkpoint-loop](autoresearch-auto-checkpoint-loop/SKILL.md)** | The background auto-commit loop for tasks running longer than 15 minutes. Provides continuous crash safety (bounded interval loss) distinct from the milestone checkpoint. |
| **[autoresearch-data-contract-validator](autoresearch-data-contract-validator/SKILL.md)** | Pre-run structural gate: asserts that the training and evaluation pipelines share a compatible (input, target) pairing contract — shape, dtype, encoding, value range. Catches the off-by-one alignment and label-encoding bugs the semantic shuffle-test does not catch. |
| **[autoresearch-modular-block](autoresearch-modular-block/SKILL.md)** | Designing a toggleable Boolean-flag composable block where each flag is one orthogonal design choice. Includes the mandatory all-flag-combinations smoke test that is the UNIT-rung gate. |
| **[autoresearch-doc-organization](autoresearch-doc-organization/SKILL.md)** | The repo-root discipline: at most 4 canonical root files, a subdir contract for every other file type, and the structure of a front-door README that surfaces both positive and negative findings with equal prominence. |
| **[autoresearch-experiment-archive](autoresearch-experiment-archive/SKILL.md)** | The per-experiment archive taxonomy: one subdirectory per experiment, mandatory detailed README, self-contained dashboard, and immutability-after-verdict rule. The unit of cold reproducibility. |
| **autoresearch-winner-archive** | Archiving a KEEP-and-new-global-best config: frozen config, full results, provenance, and the champion pointer update. |

> The supporting skills above are authored in parallel by teammates. The spine
> cross-references each of them by relative path; if a supporting skill is not
> yet present, the spine section that needs it still stands on its own.

---

## Two cross-cutting disciplines every skill assumes

Every workflow in this pack is wrapped in two project-wide disciplines that the
SKILL.md files reference but do not redefine each time:

- **Periodic GitHub checkpoint.** Commit + push on every milestone: a file edit
  with tests green, a run-folder produced, a ledger/dashboard refresh, a
  skill/constitution edit, **before AND after** every background-task launch,
  and every ~15 min of active editing — plus first thing on every wake-up. Many
  small commits beat one big commit. A power outage must never lose progress.
  Per-experiment commit happens BEFORE the next launch; if the working tree is
  dirty from the prior experiment, STOP and commit first. Never `--no-verify`,
  never `--amend`. See `autoresearch-checkpoint/SKILL.md` for the trigger table.
- **Test discipline.** Every new module/class/function ships with a unit test
  exercising shape/contract, **every Boolean-flag combination**, and the bug
  class it was written to fix. Tests must pass (Rung-0 UNIT green) before any
  background compute launches. A runner that can write a placeholder into a
  pre-run reasoning field is a bug — the test suite asserts it refuses. The
  all-flag-combinations smoke test is defined in
  [`autoresearch-modular-block`](autoresearch-modular-block/SKILL.md).

---

## Content-agnostic design rules for these skills

1. **No domain content.** A skill that names a specific dataset, model, prior,
   or phenomenon is leaking. Rewrite to the abstract role: "the project's runner
   module", "the champion config", "the primary efficacy axis", "the held-out
   benchmark at this rung". If a reader can tell what field this is for, fix it.
2. **Each skill is one workflow.** If two workflows share 80%+ of their steps,
   they are the same skill — merge them. If a skill has more than ~5 exit
   branches, split it.
3. **YAML frontmatter is mandatory.** `name` + `description` (one sentence on
   when to use). No other fields unless the skill spawns agents.
4. **Reference infra paths abstractly.** Use "the project's reasoning module" or
   "the runner contract", not a concrete source path — so the skill ports
   unchanged to the next topic.

---

## Cross-references

- The spine: [`autoresearch-meta/SKILL.md`](autoresearch-meta/SKILL.md).
- The constitution this pack generalizes lives at the project root
  (`CLAUDE.md`); the detailed inner loop lives at `AUTORESEARCH_PROCESS.md`. The
  meta pack and the project's own concrete instantiation stay in sync: a process
  improvement learned in the instantiation is ported back here.
