# Verification Rubrics — steeringresearch

> Scoreable rubrics to verify the project is complete and correct. Each item is
> PASS / PARTIAL / FAIL with concrete evidence (file path / command / commit).
> The ICML sign-off (Rubric E) is the capstone and consumes A–D.
> Maintained as the single source of truth for "are we done?".

---

## Rubric A — Original brief coverage (the user's requirements)

| # | Requirement (from the brief) | Evidence | Verdict |
|---|---|---|---|
| A1 | Read the full steering corpus first | `corpus/` (10 docs) ingested; CLAUDE.md/skills cite them | |
| A2 | Study prior autoresearch methodology (dsbench + sacgeometry + runner) | meta-skills ported from sacgeometry; AUTORESEARCH_PROCESS.md | |
| A3 | Form a CLAUDE.md for steering research | `CLAUDE.md` (16 sections) | |
| A4 | Skills for the research (experiment backbone, hill-climb, etc.) | `skills/` (14) + `meta-skills/` (27) | |
| A5 | Experiment hypothesis backbone + ladder climb | `IDEA_TABLE.md` (70 hyps), 5-rung ladder, ledger | |
| A6 | Smallest Gemma for fast iteration on 4090 | `gemma-3-270m-it` default; runs on 16GB | |
| A7 | Prepare all experiments, skills, dashboards with rigor; don't miss steps | rubrics B–E below | |
| A8 | Planning is huge; use agentic teams | 14+ agents dispatched across waves (see git log) | |
| A9 | Domain-agnostic META skill reproducing the process, audited | `meta-skills/` + `audits/META_PROCESS_AUDIT.md` | |
| A10 | Rich, transparent dashboard with sub-links to sub-dashboards | Rubric B | |
| A11 | Start doing principled experiments | exp#1–19 logged; real Gemma E3/E4/E27 | |
| A12 | High-fidelity research with ICML-reviewer-caliber audit | `audits/ICML_REVIEW.md` + Rubric E | |
| A13 | Use Gemma for everything | Gemma downloaded + default + all real runs on Gemma | |
| A14 | Elite software-engineering code quality | ruff+mypy clean, 46 tests, `audits/CODE_QUALITY.md` | |

**A passes iff** every row is PASS (PARTIAL allowed only with a tracked gap in §F).

---

## Rubric B — Dashboard (rich, transparent, hierarchically sub-linked)

| # | Criterion | How to verify | Verdict |
|---|---|---|---|
| B1 | Master dashboard exists + GitHub-Pages mirror | `dashboard/index.html` + `docs/dashboard/index.html` | |
| B2 | Sortable + type-to-filter runs table; default sort composite desc; champion highlighted | open + inspect; `data-v` sort, `#q` filter | |
| B3 | Every numeric cell carries n=X + SCREENING/EVALUATION chip | grep chips in index.html | |
| B4 | 5-axis panel (radar/parallel-coords): behavior, capability, coherence, safety, selectivity | `plot_radar.png` + `plot_parcoords.png` present | |
| B5 | Pareto panels (≥3) with prior/baseline stars + ≥1 dominated row | `plot_pareto_*.png` | |
| B6 | Ladder board (rung reached + gate + failure_reason) | section present | |
| B7 | Geometry panel (Δ‖h‖, eff-rank, norm budget) | `plot_geometry.png` | |
| B8 | Stack/compete matrix on master | grep STACK/COMPETE | |
| B9 | COMPOSITE fingerprint + commit SHA + circularity line in footer | grep `a9001e87087e` | |
| B10 | Per-hypothesis sub-dashboards exist + link from master | `ideas/<id>/dashboard/` + `docs/dashboard/hyp/` | |
| B11 | Per-experiment pages: full 7-step reasoning rendered, sweep curves, side-by-side samples | `docs/dashboard/experiments/expNNN.html`; no `##`/`**`/`\|---\|` leak | |
| B12 | 3-tier sub-linking master→hypothesis→experiment, both directions | click-through; back-links | |
| B13 | Self-contained (no CDN/JS framework); PNG not SVG; no emoji | inspect HTML | |
| B14 | Reflects ALL logged experiments (count matches experiment_log.jsonl) | row count == JSONL lines | |

**B passes iff** B1–B14 all PASS. Automated check: `scripts/verify_dashboard.py`.

---

## Rubric C — GitHub docs & reproducibility

| # | Criterion | Evidence | Verdict |
|---|---|---|---|
| C1 | README: what/why/layout/quickstart, links to dashboard + paper | `README.md` | |
| C2 | `docs/index.html` GitHub-Pages landing page → dashboard + findings + paper | `docs/index.html` | |
| C3 | Reproducibility: requirements.txt, exact run commands, seeds, model + license note | `requirements.txt`, `NEXT_STEPS.md`, `README.md` | |
| C4 | Paper draft (honest scope) in `paper/PAPER.md` | exists, ICML-format | |
| C5 | FINDINGS.md rigor-gated; SCREENING vs EXTERNAL clearly separated | `FINDINGS.md` | |
| C6 | All inherited corpus numbers tagged [NEEDS VERIFICATION] | grep | |
| C7 | No secrets committed (.hf_token gitignored; no token in history) | `git log -p \| grep hf_` = clean | |
| C8 | Every claim links to its experiment/evidence (first-mention linkification) | docs review | |
| C9 | Clean commit history, descriptive messages, co-author trailer | `git log` | |
| C10 | Optional: pushed to a GitHub remote OR push instructions documented | README / NEXT_STEPS | |

**C passes iff** C1–C9 PASS (C10 documented).

---

## Rubric D — Scientific & code rigor (internal)

| # | Criterion | Evidence | Verdict |
|---|---|---|---|
| D1 | 7-step pre-registered reasoning for every experiment (no fabricated fields) | `reasoning_annotations.json`; runner refuses placeholders | |
| D2 | Goodhart-resistant composite, SHA-256 fingerprinted, unchanged across project | `a9001e87087e` everywhere | |
| D3 | Screening (n≤3) vs evaluation (n≥7) honestly labeled; no n=1 external claims | FINDINGS, ledger tags | |
| D4 | Measurement validity: behavior non-circular (generation), safety real (generation) | `behavior_scorer=generation`, `safety_real=true` in rows | |
| D5 | Real model = Gemma; cross-model corroboration where claimed | exp#15–19 Gemma | |
| D6 | Geometry leading-indicators logged every run (offshell etc.) | row schema | |
| D7 | ruff clean · mypy clean · pytest green | `audits/CODE_QUALITY.md` + CI command | |
| D8 | Dual-track audit done (impl-critic / sci-critic / data / shuffle) at least at methodology level | audits/ | |
| D9 | Honest error reporting (no "gated" masking SSL/OOM) | `model.py` taxonomy | |
| D10 | Limitations stated prominently (synthetic data, no judge, n=1, single behavior) | FINDINGS, paper abstract | |

---

## Rubric E — ICML reviewer sign-off (capstone)

The reviewer scores the project AS WHAT IT HONESTLY IS: an **infrastructure +
methodology + screening-results** contribution, NOT a steering-claims paper.

| # | Acceptance criterion | Bar | Verdict |
|---|---|---|---|
| E1 | Reproducible harness with correct mechanics (hooks, extraction, composite) | no mechanical bugs | |
| E2 | Honest measurement instruments (non-circular behavior, real safety) | both real, tagged | |
| E3 | Rigor protocol present & followed (pre-reg, n-floor, fingerprint, ladder) | followed | |
| E4 | Findings correctly scoped to SCREENING; no overclaiming | zero external claims on n=1 | |
| E5 | Limitations in the abstract; required-experiments enumerated | present | |
| E6 | Reproducibility: anyone can rerun from README + token | yes | |
| E7 | Novelty/value of the contribution stated honestly (methods/infra, geometry findings) | yes | |
| E8 | No internal contradictions; circularity disclosed | clean | |

**Sign-off statement (target):** "ACCEPT as a methodology/infrastructure
contribution with reproducible screening results; the steering EFFICACY claims
remain correctly gated on the enumerated required-experiments (real AxBench +
calibrated judge + n≥7 + prompting baseline + multi-behavior). No overclaiming
detected." The reviewer must EITHER sign this off OR list the exact blocking items.

---

## §F — Open gaps tracker (must be empty or each gap owned)

| gap | rubric item | owner/plan | status |
|---|---|---|---|
| Real AxBench + LLM-judge behavior | D4/E2 (full) | required-experiment; documented in FINDINGS | OPEN (out of overnight scope) |
| n≥7 evaluation + prompting baseline | D3/E (full claims) | required-experiment | OPEN |
| gemma-3-1b-it standard-rung reproduction | A6/D5 | campaign C6 | IN PROGRESS |

*Rubrics are the contract. The ICML reviewer (Rubric E) signs off only when A–D
are green and §F gaps are correctly scoped as future work, not hidden.*
