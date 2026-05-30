# META autoresearch skill-pack audit — faithful-reproduction & dashboard-mandate review

**Auditor:** skeptical research-methodology auditor (same-model-family agent).
**Date:** 2026-05-30.
**Scope:** `meta-skills/` (domain-agnostic pack) vs the gold-standard
`C:\Users\evija\sacgeometry\skills\` originals + `AUTORESEARCH_PROCESS.md`,
plus `CLAUDE.md`, `AUTORESEARCH_PROCESS.md`, `skills/` (steering
instantiation), and `src/steering/dashboard.py`.

---

## Summary verdict

**The meta pack faithfully reproduces the CORE autoresearch process — the
7-step ritual, 5-rung ladder, fingerprinted composite, funnel + full
statistical-rigor floor, dual-track audit + both negative controls, the
checkpoint heartbeat, the multi-agent dispatch pattern, and the
three-tier hierarchically-linked dashboard mandate are all PRESENT and
high-fidelity.** It is NOT a complete reproduction of the original
*skill set*: of 28 original `autoresearch-*` skills, **18 are COVERED, 2
PARTIAL, 8 MISSING**. The single most serious defect is that
**`autoresearch-combo-ladder` is listed in the meta `README.md` index and
cross-referenced by the spine §6 but the directory does not exist** — a
dangling skill plus three other dead cross-references. The dashboard
*mandate* is fully specified in the SKILLs + CLAUDE.md; the `dashboard.py`
*implementation* omits three mandated surfaces (stack/compete matrix,
per-experiment sweep curves, per-experiment side-by-side samples).

- **Coverage:** COVERED 18 · PARTIAL 2 · MISSING 8 (of 28 originals).
- **Process elements (11 audited):** PRESENT 11 · WEAK 0 · ABSENT 0.
- **Dashboard richness (10 surfaces):** PRESENT 7 · WEAK 3 · ABSENT 0
  (the 3 WEAK are spec-present / impl-missing).
- **Content-agnostic leak scan:** PASS (no domain leak).

> Internal QA pass by a same-family agent; independent external review
> pending. A same-family audit is a useful filter, not an external seal.

---

## 1. Coverage matrix (original `sacgeometry/skills/autoresearch-*` → `meta-skills/`)

| # | Original skill | meta-skills counterpart | Status | Lost content / note |
|---|---|---|---|---|
| 1 | autoresearch-experiment | autoresearch-experiment | COVERED | Full 7-step ritual + word/citation gates + fingerprint + append-only + no-bypass reproduced verbatim. |
| 2 | autoresearch-reasoning-entry | (merged into autoresearch-experiment) | COVERED | Citation Rigor + Blob Completeness folded into experiment skill (legit 80%-overlap merge per design rule 2). BUT `data-split-audit` + `paper-rigor`/index still cross-reference a non-existent `autoresearch-reasoning-entry/` → dead link. |
| 3 | autoresearch-tiered-ladder | autoresearch-tiered-ladder | COVERED | All 5 rungs, gates, promotion/demotion log, same-axes invariant. Log verdict vocab is PROMOTE/DEMOTE/HOLD (vs KEEP/DISCARD elsewhere) — harmless divergence. |
| 4 | autoresearch-per-hypothesis-hillclimb | autoresearch-per-hypothesis-hillclimb | COVERED | 20–25-trial coordinate descent, strict-`>` champion rule, n≥3 confirm, per-hypothesis dashboard. Spec says "small-multiples **SVG**" — contradicts the PNG-not-SVG typography rule (minor). Cross-refs missing `ablation-sweep` + `auto-checkpoint-loop`. |
| 5 | autoresearch-paper-rigor | autoresearch-paper-rigor | COVERED | Wilcoxon + bootstrap + Holm-Bonferroni + noise band + pre-registration + verdict tiers + circularity disclosure. Tier name `UNTESTED_ON_RIGHT_SPLIT` here vs spine's `UNTESTED_ON_RIGHT_DATASET` (naming drift). |
| 6 | autoresearch-shuffle-test | autoresearch-shuffle-test | COVERED | 3 modes, PASS/WEAK/FAIL, chance-baseline derivation, provenance. Excellent. |
| 7 | autoresearch-data-split-audit | autoresearch-data-split-audit | COVERED | Full auditor suite + `audit_or_die()` + no `--bypass` gate. |
| 8 | autoresearch-dashboard | autoresearch-dashboard | COVERED | Master table + radar + Pareto + ladder board + stack/compete panel + footer. |
| 9 | autoresearch-dashboard-comprehension | autoresearch-dashboard-comprehension | COVERED | Reader's-guide + tier/`n=` chip discipline. |
| 10 | autoresearch-per-experiment-page | autoresearch-per-experiment-page | COVERED | 7-step render + sweep curves (Sec 8) + side-by-side samples (Sec 9) + CIs. (Impl gap noted in §3.) |
| 11 | autoresearch-typography-and-rendering | autoresearch-typography-and-rendering | COVERED | Rendered-markdown, PNG-not-SVG, self-contained, no-emoji. |
| 12 | autoresearch-link-discipline | autoresearch-link-discipline | COVERED | HEAD-testing absolute blob links; mandatory sub-links. |
| 13 | autoresearch-critic-team | autoresearch-critic-team | COVERED | PASS/MINOR/MAJOR/BROKEN, "be conservative with PASS", circularity disclosure, verify-the-complaint. |
| 14 | autoresearch-scicritic-team | autoresearch-scicritic-team | COVERED | NOVEL/DERIVATIVE/NUMEROLOGY/UNFALSIFIABLE/FALSIFIED + UNTESTED_ON_RIGHT_DATASET + circularity. |
| 15 | autoresearch-multi-agent-dispatch | autoresearch-multi-agent-dispatch | COVERED | Disjoint scopes + scoped add + 5-attempt retry + pull-rebase + ≤250-word returns. |
| 16 | autoresearch-checkpoint | autoresearch-checkpoint | COVERED | Trigger table, before/after background task, ≤15 min, no `--no-verify`/`--amend`, push-or-worthless. |
| 17 | autoresearch-session-resume | autoresearch-session-resume | COVERED | Crash-recovery checkpoint doc (present in dir listing). |
| 18 | autoresearch-idea-scaffold | autoresearch-idea-scaffold | COVERED | `ideas/_TEMPLATE` per-hypothesis sub-project scaffold (present). |
| 19 | autoresearch-fixer-campaign | autoresearch-fixer-campaign | COVERED | Bounded fix → re-screen → re-hill-climb (present + cross-referenced). |
| 20 | autoresearch-winner-archive | autoresearch-winner-archive | COVERED | Frozen config + provenance + champion pointer (present). |
| 21 | **autoresearch-combo-ladder** | **(none)** | **MISSING** | **Listed in meta README index + cross-referenced by spine §6, but NO directory exists.** Orthogonal combo ladder concept survives in spine §6 prose; its owning skill (stack/compete decision rule, additive 2→N ladder, everything-on-forbidden, hill-climb-before-combo) is absent. Steering pack DOES have `steering-combo-ladder`. |
| 22 | autoresearch-ablation-sweep | (none) | MISSING | The screening-sweep workflow (baselines + single-prior-on + leave-one-out + curated-60-min subset). Referenced by hillclimb as the screening counterpart → dead link. Spine names "screen 1 config" but no owning skill. |
| 23 | autoresearch-auto-checkpoint-loop | (none) | MISSING | The background auto-commit loop for >15-min tasks (distinct from the milestone `checkpoint` skill). Cross-referenced by hillclimb → dead link. Crash-safety for long sweeps lost. |
| 24 | autoresearch-data-contract-validator | (none) | MISSING | The (x_train,y_train) vs (x_val,y_val) shape/dtype/encoding contract that caught the off-by-one alignment bug. Partially overlaps shuffle-test (semantic) but the pre-run *structural* contract gate is gone. |
| 25 | autoresearch-modular-block | (none) | MISSING | Toggleable Boolean-flag composable block + the mandatory smoke-test contract. The meta `README.md` "test discipline" paragraph still points readers to this skill — but it does not exist in the pack. |
| 26 | autoresearch-doc-organization | (none) | MISSING | The ≤4-canonical-root-files rule + conference front-door README pattern + subdir contract. No counterpart. |
| 27 | autoresearch-experiment-archive | (partially winner-archive) | PARTIAL | winner-archive covers KEEP-and-new-global-best archival; the general per-experiment taxonomy directory + very-detailed-design README contract for *every* archived experiment is not reproduced. |
| 28 | autoresearch-dataset-loader | (none) | PARTIAL | Intentionally domain-infra (torchvision/HF/MedMNIST/SSL workaround). data-split-audit covers the audit half; the loader-wiring half is out of scope by the content-agnostic rule — acceptable omission but noted. |
| — | autoresearch-topology-metrics | (none) | MISSING(by design) | Domain-specific (Betti/CKA/equivariance). Correctly excluded as leaking content; not counted as a defect. |
| — | autoresearch-meta (NEW) | autoresearch-meta | n/a | New spine with no original counterpart — net addition, high quality. |

**Originals with NO counterpart:** combo-ladder, ablation-sweep,
auto-checkpoint-loop, data-contract-validator, modular-block,
doc-organization (6 genuine gaps), plus topology-metrics &
dataset-loader (excluded by the content-agnostic rule — not faults).

---

## 2. Per-element PRESENT / WEAK / ABSENT

| Element | Status | Location |
|---|---|---|
| 7-step ritual (diagnose→cite→hypothesise→predict→execute→analyse→checkpoint) + word/citation gates | PRESENT | autoresearch-meta §2; autoresearch-experiment §"the ritual" (lines 19–61); CLAUDE.md §5 |
| 5-rung ladder (UNIT/SMOKE/DEV/STANDARD/FULL) + promotion gates | PRESENT | autoresearch-tiered-ladder §"The 5 rungs" + §"Promotion-gate rule"; meta §3; CLAUDE.md §4 |
| Fingerprinted Goodhart-resistant composite (one-sided max(0,·) penalties, SHA-256, edit=BLOCKER) | PRESENT | autoresearch-meta §4; experiment §"hard rules"; CLAUDE.md §6; impl `eval.py:COMPOSITE_FORMULA` + `composite_fingerprint()` |
| Screening → hill-climb → evaluation funnel (n≤3 screen / 20–25 coord-descent / n≥7 confirm) | PRESENT | autoresearch-per-hypothesis-hillclimb (whole); meta §5; CLAUDE.md §8 |
| Statistical rigor floor: Wilcoxon + bootstrap + Holm-Bonferroni + pre-registration + verdict tiers | PRESENT | autoresearch-paper-rigor Pillars 1–3; meta §5; CLAUDE.md §7. (Empirical 2σ noise band also present.) |
| Orthogonal combo ladder (stack-only-orthogonal, additive 2→N, everything-on forbidden) | PRESENT (concept) / ABSENT (owning skill) | Concept fully in autoresearch-meta §6 + CLAUDE.md §9 + skills/steering-combo-ladder. **The dedicated `meta-skills/autoresearch-combo-ladder/SKILL.md` does NOT exist** despite being indexed/cross-referenced. |
| Checkpoint heartbeat (milestone trigger table, before/after bg task, no --amend/--no-verify) | PRESENT | autoresearch-checkpoint (trigger table + rules); meta §11-wrap; CLAUDE.md §13 |
| Multi-agent dispatch retry pattern (disjoint scope, scoped add, 5-attempt retry, pull-rebase, bounded return) | PRESENT | autoresearch-multi-agent-dispatch (whole); meta §8; CLAUDE.md §14 |
| Impl-critic dual-track leg | PRESENT | autoresearch-critic-team (PASS/MINOR/MAJOR/BROKEN); meta §10.1 |
| Sci-critic dual-track leg (NUMEROLOGY/UNFALSIFIABLE/HARKed) | PRESENT | autoresearch-scicritic-team; meta §10.2 |
| Shuffle-test negative control | PRESENT | autoresearch-shuffle-test (3 modes); meta §10.4 |
| Data-split audit (`audit_or_die`, no --bypass) | PRESENT | autoresearch-data-split-audit; meta §10.3 |
| No-self-grading circularity disclosure ("Internal QA pass — external review pending") | PRESENT | critic-team + scicritic-team + paper-rigor Pillar 5 + meta §8 + dashboard.py `_footer()` emits it on every page |

All 11 mandated process elements are PRESENT. The only caveat is that
the **combo-ladder is present as spine prose but its owning skill file is
absent** — the *process step* exists, the *skill* does not.

---

## 3. Dashboard richness audit (user's explicit priority)

Spec sources: autoresearch-meta §7, autoresearch-dashboard,
dashboard-comprehension, per-experiment-page, CLAUDE.md §11.
Impl source: `src/steering/dashboard.py`.

| Mandated surface | Spec | Impl (`dashboard.py`) | Status |
|---|---|---|---|
| Master dashboard, sortable + type-to-filter table, default sort composite desc, champion highlighted | YES (meta §7A) | YES — `render_master` + `SORT_SCRIPT` + `.champion` row | PRESENT |
| `n=X` + SCREENING/EVALUATION chip on EVERY numeric cell | YES (dashboard-comprehension) | YES — `_num_cell()` emits seed-badge + tier chip | PRESENT |
| 5-axis radar / parallel-coordinates per method | YES (meta §7A) | YES — `plot_radar` + `plot_parcoords` (behavior/capability/coherence/safety/selectivity) | PRESENT |
| Pareto panels with dominated rows visibly marked | YES | YES — `plot_pareto` ×3 (capability/coherence/safety), dominated → red outline, baselines → stars | PRESENT |
| Ladder board (rung reached + gate + failure_reason) | YES | YES — `ladder_board()` + master table | PRESENT |
| Geometry / leading-indicator panel | YES (CLAUDE.md §3,§11C) | YES — `plot_geometry` (Δ‖h‖, eff-rank drop, norm budget) | PRESENT |
| COMPOSITE fingerprint footer (+ git SHA + circularity line) | YES (meta §4,§7) | YES — `_footer()` prints formula + `composite_fingerprint()` + SHA + "Internal QA pass — external review pending" | PRESENT |
| **Stack/compete matrix panel on master** | YES (meta §7A, dashboard §Panel 5, CLAUDE.md §11A) | **NO — `render_master` renders runs/radar/Pareto/ladder/geometry only; no stack/compete matrix** | **WEAK (spec-present, impl-missing)** |
| **Per-experiment sweep curves** | YES (per-experiment-page Sec 8; CLAUDE.md §11C "α/layer sweep curves") | **NO — `render_experiment` has reasoning/config/axis/geometry/composite-breakdown but NO sweep-curve plot** | **WEAK (spec-present, impl-missing)** |
| **Per-experiment side-by-side generation samples (steered vs unsteered)** | YES (per-experiment-page Sec 9; CLAUDE.md §11C) | **NO — `render_experiment` renders no sample-comparison section** | **WEAK (spec-present, impl-missing)** |
| 3-tier sub-linking: master → per-hypothesis → per-experiment (with rendered 7-step) | YES (meta §7 diagram) | YES — `build_all_dashboards` writes master + `ideas/<dir>/dashboard/` + `experiments/expNNN.html`; `render_experiment` renders 7-step via `md_to_html` (no `##`/`**`/`|---|` leak); back-links wired both directions; docs/ mirror produced | PRESENT |

**Net:** the mandate is fully *specified* (rich + transparent +
hierarchically sub-linked). The *implementation* delivers 7 of 10 named
surfaces plus the full 3-tier link graph; it is missing the **stack/compete
matrix** on the master and the **sweep-curve** + **side-by-side-sample**
sections on the per-experiment leaf — all three are required by the SKILL
spec and CLAUDE.md but absent from `dashboard.py`. The per-experiment page
adds a (non-mandated, welcome) composite-breakdown table.

---

## 4. Content-agnostic leak scan

Scanned `meta-skills/` for `steering | Gemma | refusal | activation |
JailbreakBench | Rogue Scalpel | RTX 4090 | MMLU | residual stream`
(case-insensitive).

**Result: PASS — no domain leak.** The only hits are legitimate
generic/illustrative English, all permitted by README design rule 1:

- `autoresearch-meta/SKILL.md:176` — "over-refusal" as a *generic*
  example cost axis in a code comment (`# e.g. selectivity / over-refusal`).
- `autoresearch-meta/SKILL.md:458` — "placeholder-refusal" (runner
  refusing to launch), generic.
- `autoresearch-data-split-audit/SKILL.md:205` — "refusal message",
  generic (the runner's refusal text).
- `autoresearch-tiered-ladder/SKILL.md:59` — "an activation vector
  changes a residual stream" offered explicitly as *one example among
  several* ("a mask changes the loss, a weight delta changes the logits").
- `autoresearch-tiered-ladder/SKILL.md:144` — "safety and refusal
  benchmarks (if applicable)", generic + hedged.

None name the steering topic, Gemma, the 5 steering axes, or any
domain-locked content. The pack is content-agnostic.

---

## 5. Prioritized gaps to fix

1. **[BLOCKER] Create `meta-skills/autoresearch-combo-ladder/SKILL.md`.**
   It is indexed in the README and cross-referenced by the spine §6 but
   does not exist — a dangling skill that breaks the orthogonal-combo
   element's "owning skill" promise. Port from `skills/steering-combo-ladder`
   with domain content stripped (stack/compete decision rule, additive
   2→N ladder, everything-on-forbidden, hill-climb-before-combo).

2. **[MAJOR] Fix the four dead cross-references.** `data-split-audit` and
   the index point to `autoresearch-reasoning-entry/` (merged away);
   `per-hypothesis-hillclimb` points to `autoresearch-ablation-sweep/` and
   `autoresearch-auto-checkpoint-loop/`; the README "test discipline"
   paragraph points to `autoresearch-modular-block/`. Either create these
   skills or repoint the links (reasoning-entry → experiment; modular-block
   discipline → inline in README).

3. **[MAJOR] Close the dashboard SKILL-vs-impl gap in `dashboard.py`.**
   Implement (a) the **stack/compete matrix** panel in `render_master`
   (spec: dashboard §Panel 5 / meta §7A / CLAUDE.md §11A), (b) the
   **sweep-curve** plot and (c) the **side-by-side steered-vs-unsteered
   sample** section in `render_experiment` (spec: per-experiment-page
   Sec 8 & 9 / CLAUDE.md §11C). All three are mandated and currently
   absent, leaving the leaf page and master incomplete against the mandate.

4. **[MINOR] Restore the missing structural-integrity & infra skills as
   process steps.** `data-contract-validator` (the pre-run (x,y) pairing
   contract — a *structural* gate the shuffle-test's *semantic* check does
   not replace) and `auto-checkpoint-loop` (background crash-safety for
   long sweeps) are genuine process losses. `doc-organization` (≤4 root
   files) and the general `experiment-archive` taxonomy are lesser losses.
   Decide consciously whether each is in-scope for the portable pack;
   if out, remove its references so the pack is internally consistent.

5. **[MINOR] Resolve naming/format drifts.** (a) verdict-tier name
   `UNTESTED_ON_RIGHT_SPLIT` (paper-rigor) vs `UNTESTED_ON_RIGHT_DATASET`
   (spine §5 / scicritic) — pick one. (b) hillclimb spec says per-axis
   Pareto "small-multiples **SVG**" — contradicts the PNG-not-SVG
   typography hard rule; change to PNG. (c) ladder-log verdict vocab
   PROMOTE/DEMOTE/HOLD vs the experiment KEEP/DISCARD/NEAR-MISS — note the
   mapping so the dashboard's `ladder_board()` (which reads `status ==
   "KEEP"`) and the ladder-log schema agree.

---

## Circularity disclosure

This is an **internal QA pass by a same-model-family agent; independent
external review is pending.** The verdicts above (COVERED/PARTIAL/MISSING,
PRESENT/WEAK/ABSENT) are a useful filter for completeness and faithfulness,
not an external seal of approval. The pack's own paper-rigor Pillar 5 and
critic-team disclosure apply to this audit as well: until the audit
protocol is calibrated against a known-good external reference, treat
these findings as descriptive, not diagnostic.

---

## 6. Resolution log (fix campaign 2026-05-30)

All audit gaps closed by a 2-agent fixer campaign (disjoint scopes: docs vs
dashboard impl), then verified-the-complaint:

| Gap | Severity | Resolution | Verified |
|---|---|---|---|
| `autoresearch-combo-ladder` missing | BLOCKER | Created (generalized from sacgeometry). | file exists ✓ |
| 4 dead cross-references | MAJOR | reasoning-entry → experiment; ablation-sweep / auto-checkpoint-loop / modular-block created. | grep = 0 dead refs ✓ |
| dashboard 3 surfaces (stack/compete matrix, sweep curves, side-by-side samples) | MAJOR | Implemented in `dashboard.py` + tests. | master shows STACK/COMPETE; 32 tests pass ✓ |
| Missing infra skills (ablation-sweep, auto-checkpoint-loop, data-contract-validator, modular-block, doc-organization, experiment-archive) | MINOR | All 6 created as generalized ports. | 27 SKILL.md total ✓ |
| Naming drifts (SPLIT→DATASET, SVG→PNG, ladder verdict mapping) | MINOR | Renamed/repointed; PROMOTE↔KEEP mapping noted. | grep = 0 SPLIT / 0 hillclimb-SVG ✓ |

**Post-fix state:** coverage 25 COVERED / 1 PARTIAL (dataset-loader, out of scope
by content-agnostic rule) / 0 BLOCKER; 11/11 process elements PRESENT; dashboard
10/10 mandated surfaces PRESENT; leak scan PASS; `pytest tests/` = 32 passed.
Circularity disclosure unchanged: internal QA pass, external review pending.
