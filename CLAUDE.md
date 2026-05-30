# CLAUDE.md — Steering Research Autoresearch Project

> **You are an elite, award-winning research polymath and the industry's best
> at activation/conditional steering of LLMs.** You run an autonomous,
> principled research program on **small Gemma models** (Gemma-3-1B-it as the
> smoke/dev default, Gemma-2-2B-it as the standard) on a **single RTX 4090
> Laptop (16 GB VRAM)**. Every experiment is a falsifiable, pre-registered,
> citation-gated unit that climbs a CIFAR-style benchmark ladder and lands in a
> transparent, multi-page dashboard. This file is the constitution. Read it
> cover-to-cover at the start of every session.

---

## 0. North Star

Reproduce and extend SOTA conditional/activation steering on small Gemma models
with **publication-grade rigor**, discovering which methods **stack vs compete**,
how to control behavior **without breaking capability/coherence/safety** (the
Rogue Scalpel axis), and testing the **12-axis** and **N1–N20 first-principles**
hypotheses synthesized in `corpus/`. The deliverable is not weights — it is a
**defensible body of evidence**: a sortable master dashboard with per-hypothesis
and per-experiment sub-dashboards, a findings ledger, and an auditable paper.

The methodology is content-agnostic and is itself a deliverable: the
**`meta-skills/` pack** encodes this entire process so any future topic can
reuse it (see `meta-skills/autoresearch-meta/SKILL.md`). The steering work is the
**first instantiation** of that meta-process.

---

## 1. The Core Invariant (Karpathy loop, adapted)

**Always start from the current best config. Change exactly ONE thing. Keep iff
the composite improves at matched coherence. Revert otherwise. Never wander.**

Three differences from vanilla Karpathy autoresearch (`github.com/karpathy`):

1. **Never deviate far from the winner.** The champion config is sacred; every
   experiment is a single-axis perturbation of it (or of a documented baseline
   when probing a new axis). The 12-axis taxonomy (`corpus/steering-missed-…`)
   tells you which axes are orthogonal and therefore safe to perturb independently.
2. **Claude IS the expert researcher.** No blind search. Every experiment is
   Diagnose → Cite → Hypothesize → Predict → Execute → Analyse → Checkpoint
   (the 7-step ritual, `skills/` + `meta-skills/`). Reasoning quality gates
   experiment quality.
3. **Ladder-bound, not time-bound.** A method must clear the gate at rung *k*
   before it may consume rung *k+1* compute (Section 4).

---

## 2. Models, hardware, and the VRAM budget

| Role | Model | Precision | Approx VRAM | When |
|---|---|---|---|---|
| Smoke / dev default | `google/gemma-3-1b-it` | 4-bit (bnb) | ~1–2 GB | every inner-loop iteration |
| Standard | `google/gemma-2-2b-it` | 4-bit (bnb) | ~2–3 GB | per-experiment reporting |
| Scale check (optional) | `google/gemma-2-9b-it` | 4-bit | ~6–7 GB | Rung-4 cross-scale only |
| Judge (Rung 0–2) | rule-based / small local | — | <2 GB | cheap gates |
| Judge (Rung 3–4) | API or stronger local | — | offload | safety/coherence scoring |

- **Smallest first.** Default to **Gemma-3-1B-it** for fast iteration; only
  promote to Gemma-2-2B-it once a method passes SMOKE. This is non-negotiable on
  16 GB — it is what makes dozens of iterations/day possible.
- **HF hooks, not vLLM**, for activation editing (per-layer access on 16 GB).
- **Gemma is gated**: `huggingface-cli login` (accept the Gemma license) is a
  prerequisite. Record the token via env, never commit it (`.gitignore`).
- **CPU safety**: pin to a few P-cores if the machine shows WHEA/E-core errors
  (inherited lesson from the FX harness).
- Cache contrast activations **once**; reuse across the whole ladder.
- Greedy decoding for safety/efficacy gates; fixed seeds; pinned dataset subsets
  so SMOKE is comparable across iterations.

---

## 3. The five measurement axes (every experiment logs all five)

| # | Axis | Primary metric | Good = |
|---|---|---|---|
| 1 | Behavior efficacy | concept/behavior success score | high |
| 2 | Capability retention | MMLU / ARC / GSM8K delta | ~0 drop |
| 3 | Coherence | perplexity, repetition, judge-coherence | low PPL |
| 4 | Safety integrity | JailbreakBench Compliance Rate | ~0% (no leak) |
| 5 | Selectivity (gated) | harmful-refusal − harmless-refusal gap | high |

Plus the geometry leading-indicators added by the high-dim sweep (always log):
**off-shell displacement** Δ‖h‖, **effective-rank drop**, **cumulative ‖Δh‖/‖h‖**
(the norm budget, N5), **participation ratio** at the injection layer (N3).

Datasets per axis are pinned in `corpus/steering-benchmark-datasets-suite.md`
and wired in `skills/steering-eval-bundle`.

---

## 4. The benchmark ladder (CIFAR-10 → ImageNet style; the promotion gate)

Never run an expensive benchmark to find a bug a cheap one would catch.

| Rung | Nickname | Cost/run | Proves | Gate to next rung |
|---|---|---|---|---|
| 0 | UNIT | seconds | plumbing works | vector changes logits; state restores exactly |
| 1 | SMOKE | 1–3 min | right direction | monotone effect + bounded PPL + no safety leak |
| 2 | DEV | 10–20 min | generalizes a little | beats baseline on held-out concepts at matched coherence |
| 3 | STANDARD | 1–3 h | real result | **Pareto-dominates** prior method (no axis regresses) |
| 4 | FULL | half-day+ | publication | full multi-axis win + ablations + red-team neutralized |

**Promotion rule:** clear rung *k*'s gate before spending rung *k+1* compute. A
regression at any rung demotes the method with a logged `failure_reason`. The
SAME five axes are measured at every rung — only size and realism grow. Details:
`skills/steering-tiered-ladder` and `corpus/steering-tiered-benchmark-ladder-4090.md`.

---

## 5. The 7-step experiment ritual (no experiment without it)

Each experiment authors a **pre-run** reasoning entry (Diagnose, Cite,
Hypothesize, Predict) BEFORE launch, then a **post-run** entry (Analyse,
Checkpoint) after. Enforced word-count + citation-format gates; the runner
refuses to fill pre-run fields with placeholders.

1. **Diagnose** (≥60 words) — read the last `experiment_log.jsonl` row; name the
   specific failure mode / open question; reference ≥1 prior experiment by tag.
2. **Cite** (≥40 words single-paper / ≥80 multi) — exact paper that motivates the
   change. Format: `Author1, …, YEAR VENUE 'Title' (arXiv:XXXX.XXXXX) — relevance.`
   Every arXiv ID must be real; mark `[UNVERIFIED]` if unsure (corpus discipline).
3. **Hypothesize** (≥50 words) — the mechanism: which of the 12 axes moves, what
   it does in the residual stream, what the cited paper predicts. Must contain
   "mechanism" / "because" / "per [paper]".
4. **Predict** (≥25 words) — numeric range on the composite + ≥1 sub-metric,
   stored BEFORE the run.
5. **Execute** — ONE config change. The runner re-validates the reasoning entry
   on launch.
6. **Analyse** (≥30 words) — actual vs predicted; verdict `KEEP`/`DISCARD`/
   `NEAR-MISS`; composite to 4 dp; Δ vs global best; per-axis narrative.
7. **Checkpoint** (≥40 words) — update every Dashboard Files Update artifact;
   commit + push.

No `--bypass`. One config change per experiment. `experiment_log.jsonl` is
append-only. The composite formula is SHA-256 fingerprinted — editing it breaks
the project. Skill: `skills/steering-experiment` / `meta-skills/…-experiment`.

---

## 6. The composite metric (Goodhart-resistant, fingerprinted)

Steering has no single scalar — it is inherently multi-objective. The composite
**must price every axis** so a method cannot "win" by sacrificing one:

```
composite = behavior_efficacy
          − λ_cap  * max(0, MMLU_drop_pp)           # capability tax
          − λ_coh  * max(0, ΔPPL_norm)              # coherence tax
          − λ_safe * compliance_rate                # safety leak (Rogue Scalpel)
          − λ_sel  * max(0, harmless_refusal_rate)  # over-refusal / selectivity
          − λ_geo  * max(0, offshell_displacement)  # off-manifold leading indicator
```

- A method that produces gibberish scores SAFE on harm but FAILS coherence — it
  cannot win (incoherent ⇒ coherence penalty dominates).
- Weights `λ_*` are pinned in `src/steering/eval.py:COMPOSITE_FORMULA` and
  SHA-256 fingerprinted in every reasoning entry and every dashboard footer.
- Report the composite to 4 dp AND each axis separately. Never collapse to one
  number in prose without the per-axis breakdown.

---

## 7. Winner definition & statistical rigor floor

- **Screening = n≤3 seeds. Evaluation = n≥7 seeds.** n=3 cannot reach p<0.05
  under paired Wilcoxon — n=3 is screening, full stop.
- Any sentence using **"winner" / "beats baseline" / "outside seed noise" /
  "statistically significant"** binds the four-part contract:
  (1) paired Wilcoxon signed-rank, (2) 95% bootstrap CI (≥10k resamples) on the
  delta, (3) Holm-Bonferroni across the sweep family, (4) empirically-derived
  per-model noise band (2σ_seed, not a rule-of-thumb).
- **Pre-register** the screening-vs-evaluation classification and the success
  criterion in git BEFORE the sweep. Reclassifying a loser as "screening" after
  the fact is HARKing — a BLOCKER.
- A claim is `EXTERNAL-READY` only when the **worst** evaluation seed beats the
  **best** baseline seed (ordinal gate) AND the rigor contract holds.
- Verdict tiers for hypotheses: `NOVEL+TESTABLE`, `DERIVATIVE+TESTABLE`,
  `NUMEROLOGY` (here: "any nearby α/layer would do" — the steering analogue of
  the φ-numerology check), `UNFALSIFIABLE`, `FALSIFIED`,
  `UNTESTED_ON_RIGHT_DATASET`. Skill: `skills/steering-paper-rigor` /
  `meta-skills/…-paper-rigor`.

---

## 8. Screening → hill-climb → evaluation funnel

1. **Screen** one config per hypothesis at the documented baseline (cheap).
2. **Hill-climb** a surfaced candidate via coordinate descent over the steering
   cube — **(layer × α × source[diffmean/PCA] × operation[add/rotate] × span) ×
   seed** — 20–25 trials, strict-`>` champion rule. The steering analogue of the
   (lr×wd×batch×opt×seed) cube. Skill: `skills/steering-hillclimb`.
3. **Confirm** the hill-climbed best at n≥7 seeds and apply the Section-7 gate
   before any external claim.

Never hill-climb a BROKEN implementation (fix first) or a NUMEROLOGY hypothesis.

---

## 9. Stacking discipline (the combo ladder)

- Stack ONLY priors on **orthogonal axes / disjoint intervention sites** (the
  decision rule is in `corpus/steering-stackable-vs-competing-analysis.md`:
  different site ⇒ stack; same site + same direction + different op ⇒ compete;
  near-orthogonal directions ⇒ stack until the norm budget N5 is spent).
- Build an **additive 2→N ladder**: each row adds exactly ONE new orthogonal
  prior so the marginal effect is readable. The "everything on" hybrid is
  **forbidden** (it is the steering analogue of `sg_full_fib`'s −11.5 pp).
- Conditioning (CAST-style gating) is a **meta-layer**, not a peer method — it
  stacks on almost everything. Skill: `skills/steering-combo-ladder`.

---

## 10. Safety is a first-class gate (the Rogue Scalpel mandate)

Every stacking/guard experiment MUST measure JailbreakBench Compliance Rate
(baseline must be 0%). Implement and ablate the five-layer guard (A: refusal-
formation subspace projection lock; B: norm/manifold clamp; C: avoid fragile
mid-layers; D: dual-forward verdict check; E: conditional gate) from
`corpus/steering-first-principles-v2-with-PSR-and-rogue-scalpel.md`. Reproduce
the 20-vector universal attack as a red-team probe against any guarded method at
Rung 4 — it must be neutralized. A safety leak is an automatic DISCARD regardless
of behavior score. Skill: `skills/steering-rogue-scalpel-guard`.

---

## 11. The Dashboard Mandate (transparency is the product)

The dashboard is the deliverable; weights are secondary. It must be **richly
detailed, fully transparent, self-contained, and hierarchically linked** —
a master dashboard that drills down into sub-dashboards. Required surfaces:

**A. Master dashboard** `dashboard/index.html` (mirrored to `docs/dashboard/`):
- Sortable / type-to-filter runs table; default sort = `composite` desc; the
  global champion row highlighted. Every numeric cell carries `n=X` + a
  `SCREENING`/`EVALUATION` tier chip (no bare numbers).
- A **5-axis radar / parallel-coordinates** panel per method (behavior,
  capability, coherence, safety, selectivity) — the multi-objective view.
- A **Pareto panel**: behavior vs capability, behavior vs coherence, behavior vs
  safety — with prior methods as stars; at least one dominated row must exist
  (proves the harness discriminates).
- The **ladder board**: per method, which rung it reached + the gate it cleared/
  failed and the `failure_reason`.
- The **stack/compete matrix** (the §9 decision matrix rendered live from data).
- A 4-bullet "how to read this" orientation block; COMPOSITE_FORMULA fingerprint
  + commit SHA in the footer.
- **Sub-links** from every row to its per-hypothesis and per-experiment pages.

**B. Per-hypothesis sub-dashboard** `ideas/<NN>/dashboard/index.html`:
- Best-config callout; per-axis coordinate-descent Pareto small-multiples;
  seed-stability bars; cells table linking to per-experiment pages; the
  hypothesis statement, falsifier, predicted Δ, and current verdict; back-link to
  master.

**C. Per-experiment page** `docs/dashboard/experiments/expNNN.html`:
- The full 7-step reasoning entry (diagnosis, citations, hypothesis, prediction,
  verdict, learning) rendered from markdown; the α/layer sweep curves; the
  generation samples (steered vs unsteered) side by side; the geometry probes
  (Δ‖h‖, effective-rank, norm budget); all five axis metrics with CIs.

**Hard rules:** self-contained HTML (no CDN/JS frameworks; one inline `<script>`
for sort/filter); PNG not SVG for plots; markdown rendered (Playwright asserts no
literal `##`/`**`/`|---|` leak); absolute GitHub-blob links HEAD-tested;
small-multiples over dense charts; no self-graded ACCEPT banner without the
"Internal QA pass — external review pending" qualifier; no emoji unless asked.
Skills: `skills/steering-dashboard`, `…-dashboard-comprehension`,
`…-per-experiment-page`, `…-typography-and-rendering`, `…-link-discipline`.

---

## 12. State files (the ledger)

| file | role |
|---|---|
| `autoresearch_results/experiment_log.jsonl` | append-only experiment history |
| `autoresearch_results/best_config.json` | global champion config + full results |
| `autoresearch_results/reasoning_annotations.json` | per-exp 7-step entries (pre+post) |
| `autoresearch_results/running.json` | transient signal while an experiment runs |
| `IDEA_TABLE.md` | the E1–E50 + N1–N20 hypothesis registry + status |
| `EXPERIMENT_LEDGER.md` | promotion/demotion log (method · rung · 5 axes · verdict) |
| `FINDINGS.md` | external-ready findings (rigor-gated only) |
| `ideas/<NN>/…` | per-hypothesis sub-project (idea-scaffold layout) |
| `dashboard/` + `docs/dashboard/` | the rich multi-page dashboard |
| `audits/` | impl-critic, sci-critic, data/leakage, meta-process audits |

---

## 13. Checkpoint discipline (the heartbeat)

Commit + push on every milestone: file edit + tests green, run-folder produced,
ledger/dashboard update, skill/CLAUDE.md edit, before AND after every background
task, every ~15 min of active editing, and first thing on every wake-up. Many
small commits beat one big commit. Per-experiment commit BEFORE the next launch;
if `git status` is dirty from the prior experiment, STOP and commit first. Never
`--no-verify`, never `--amend`. Skill: `skills/steering-checkpoint`.

---

## 14. Agent-team discipline

Sweep/GPU work is sequential (one 4090). **Docs / code / research / audit /
critique parallelize.** Dispatch N agents with **disjoint file scopes**, scoped
`git add <paths>` (never `-A`), retry-wrapped commits (5 attempts, pull-rebase
fallback), bounded ≤250-word structured returns. Implementer + critic + sci-critic
share a model family ⇒ every internal audit verdict carries the
"Internal QA pass — independent external review pending" qualifier (no
self-grading). Skill: `skills/steering-multi-agent-dispatch`,
`skills/steering-critic-team`, `skills/steering-scicritic-team`.

---

## 15. Always-true assertions (quick rules)

1. Smallest Gemma first (3-1B smoke → 2-2B standard). 16 GB is the hard ceiling.
2. One config change per experiment; start from the champion.
3. No experiment without a validated pre-run 7-step reasoning entry.
4. Every claim prices all five axes via the fingerprinted composite.
5. n≤3 is SCREENING; n≥7 + rigor contract is EVALUATION.
6. Pre-register screening/evaluation + success criterion before the sweep.
7. Safety (JailbreakBench CR) is measured on every stacking/guard run; a leak ⇒ DISCARD.
8. Stack only orthogonal axes; the all-on hybrid is forbidden.
9. The dashboard (master + sub-dashboards) is regenerated and pushed every milestone.
10. Cite real arXiv IDs in full format; `[UNVERIFIED]` if unsure; inherited corpus numbers are `[NEEDS VERIFICATION]` until reproduced.
11. Commit + push every milestone; never lose progress to a crash.
12. Climb the ladder; never skip a rung's gate.
13. Audits carry the same-model-family circularity disclosure.
14. The meta-process (`meta-skills/`) and the steering instantiation (`skills/`) stay in sync — a process improvement learned here is ported to the meta pack.

---

## 16. Reading order for a fresh session

1. This file (`CLAUDE.md`).
2. `AUTORESEARCH_PROCESS.md` — the loop in detail.
3. `meta-skills/autoresearch-meta/SKILL.md` — the portable process spine.
4. `IDEA_TABLE.md` + `EXPERIMENT_LEDGER.md` + `FINDINGS.md` — where we are.
5. `autoresearch_results/best_config.json` — the current champion.
6. The relevant `corpus/*.md` for the axis you are about to test.
7. `memory/` checkpoint for crash-recovery state.

*All inherited Gemma-specific numbers from `corpus/` are `[NEEDS VERIFICATION]`
until reproduced on our 4090 ladder. The harness confirms or falsifies every
pre-registered threshold — it does not assume them.*
