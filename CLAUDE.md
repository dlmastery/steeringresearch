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

---

## 17. The `steering_tutorials/` pedagogical track (auxiliary deliverable)

Parallel to the autoresearch program, `steering_tutorials/` is a **standalone,
progressively harder tutorial series** that teaches activation steering from
scratch — self-contained code a newcomer can run, deliberately **independent of
the research harness** (`src/steering`). Lessons and design rules:

- **`hello_world/`** — lesson 1, the **READ** side: a 3-layer MLP probe on frozen
  Gemma-3-1B (the abliterated `DavidAU/…-heretic-…` build) layer-12 mean-pooled
  activations, classifying harmful vs benign (JailbreakBench; scaled up with the
  principled `lmsys/toxic-chat` loader). Kept **minimal** — one fixed layer, one
  fixed MLP. Rigor that *validates* the result lives here (full 12-metric suite,
  5-fold CV with 95% CIs, leakage/confound audit, OOD transfer). **Optimization
  does not** — a sweep is no longer "hello world."
- **`probe_tuning/`** — the **layer sweep** and **MLP hyperparameter sweep** live
  here, shelled out of `hello_world`. Model/config selection is by
  cross-validation, never test-set peeking.
- **`hello_world_steering/`** — lesson 2, the **WRITE** side: a CAA / diff-of-means
  refusal steering vector, applied **conditionally** via lesson-1's probe as the
  gate (CAST-style), validated by the **same Gemma** as a REFUSAL/COMPLIANCE/
  GIBBERISH judge. This is the READ→WRITE composition.
- **`reft_r1/`** — lesson 3, the **GENERATE** side: AxBench's learned rank-1 ReFT
  intervention (train r,w,b, LLM frozen) plus the honest ReFT-r1-vs-DiffMean-vs-
  prompting bake-off (AxBench: simple baselines are strong). *(Replaces an earlier
  hypernetwork draft, `hypersteer/`, retired to git history. Source: arXiv
  2501.17148 + 2404.03592.)*
- **`flas/`** — lesson 3b, **GENERATE+**: flow-based activation steering — a
  concept-conditioned velocity field integrated over a flow (flow-time = a
  continuous strength dial). (github.com/flas-ai/FLAS.)
- **Further lessons** (`multi_intent`, `rogue_scalpel`, `realignment`, `stacking`,
  and the planned CONTROL/CERTIFY/PROVE tiers) are catalogued in
  `steering_tutorials/README.md` — the course map with all lesson plans.

**Track standards (elite-data-scientist bar):** principled dataset sampling
(prompt-level labels not response-level; harm-category stratification; dedup +
group-aware splits; natural base rate reported), the full classification suite
with CIs, a leakage/length-confound audit on every dataset, honest OOD reporting
(degradations stated as prominently as wins), calibration/reliability curves, and
fixed seeds with the scaler fit on train only. Built by **disjoint-scope parallel
agent teams**; the single 4090 **serializes all GPU work** (one GPU agent at a
time) while docs/data/audit/code agents parallelize. Binary artifacts
(`probe.pt`, `features.npz`) are force-added so each lesson reproduces from the
repo.

**HARD DATA & RIGOR RUBRIC — user-mandated, NON-NEGOTIABLE (do not forget this):**

1. **≥500 positives AND ≥500 negatives per class.** Every binary lesson loads
   `common.data` at `N_PER_CLASS >= 500`; the build/extract split is `>= 300`/class
   and the eval a substantial held-out slice. **NO tiny datasets** — 30/30, 50/class,
   `N_EVAL=5/20/40/60` are FORBIDDEN as headline numbers. When a lesson needs several
   disjoint slices (extract+decomp+write, or disjoint attack halves), raise
   `N_PER_CLASS` so all slices fit (e.g. 600), or state the pool cap honestly.
2. **Concept/agent lessons are pool-limited** (toxic-chat categories cap ~100–388;
   Attack_600 has 600 → ~300/class for disjoint-split conditions). Maximize within
   the pool and **say so explicitly**; NEVER build a per-category/per-concept detector
   from `< 30` examples (`MIN_CE_EXAMPLES >= 30`, `N_EVAL_PER_CONCEPT >= 30`).
3. **Off-family judge for ALL reported numbers** — `STEER_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct`.
   The 1B model grading its own output inflates refusal; never headline a self-judged
   number.
4. **Cite every referenced arXiv paper with VERY CLEAR detail, in every lesson.**
   Each README opens with a `> **Reference:**` block listing the full paper TITLE as
   a clickable `arxiv.org/abs/XXXX.XXXXX` link **plus authors + venue/date + a
   one-line relevance note**; the top-level `steering_tutorials/README.md` theme
   tables carry a linked **Reference paper** column (full titles, every paper the
   lesson uses). Every id is WebFetch-VERIFIED before it ships (real title+authors);
   mark `[UNVERIFIED]` only if a fetch fails. `AUDIT.md` per lesson re-audits each id
   and states plainly what is reproduction vs inspired-by. Never cite from memory.
5. **Use real released benchmarks (HuggingFace) as an OOD test where they exist**
   (e.g. `intrinsec-ai/cstm-bench`, `ScaleAI/mhj`, `SafeMTData`); when the benchmark is
   small, still construct the `>= 500/class` MAIN train/eval set from available data and
   report the real benchmark as OOD.
6. **Small-N findings are PROVISIONAL.** Re-validate at `>= 500/class`; several
   findings flipped with more data (gavel calibration artifact, fine_grained null →
   positive, talan adapter-vs-rank-1 tie, non-ident recipe convergence). More extract
   data → a better direction → the finding sharpens/corrects. Never ship a small-N
   number as settled.
7. **Length/confound-match the negatives.** A detection lesson's benign hard-negatives
   must be rendered the SAME way as the positives so raw length/token count can't
   separate the classes. Run `confound_report` (length_auc / count_auc) on every
   detection lesson and report the RESIDUAL honestly; claim only the margin ABOVE the
   larger of {baseline, confound}. (Lesson: biencoder_guard benign pool drawn
   prompt-only vs positives' prompt+response gave length_auc 0.72 → fixed to 0.52 by
   drawing benigns from the same source rendering.)
8. **The trajectory/guardrail-detection family** now spans turns → tokens → agents →
   many-traces → policy-matching: `multiturn_jailbreak`, `trajguard`,
   `cross_trajectory`, `meerkat` (clustering: arXiv:2604.11806), `biencoder_guard`
   (EmbeddingGemma dual-tower: GLiNER bi-encoder 2602.18487, GLiNER Guard 2605.05277,
   Opir 2605.29659, GLiGuard 2605.07982) with the 2026 hard-negative synthesis line
   (ECIsem 2603.20990, ARHN 2604.11092, CausalNeg 2606.01304). Detection lessons take
   NO generation judge; pre-register a falsifier per claim; use a real HF benchmark
   (CSTM-Bench) as OOD.
9. **Big packages use the spine-anchor multi-team pattern.** The lead writes the FIXED
   spine first (Pydantic data models + Protocol interfaces + the config anchor + a
   safety/authorization gate), verifies it imports, then fans out disjoint-scope agents
   that build ONLY to those signatures. Relay cross-file interface notes between agents
   via `SendMessage`; the lead owns the anchor and fixes anchor bugs centrally (agents
   never edit it, never run git). Parallelize maximally: docs/code/data/test agents run
   concurrently; only GPU work serializes on the one 4090.

**`auto-redteam/` (standalone package) — authorized-research red-team harness.** An
ablated local Gemma attacker vs a pluggable frontier defender (Gemini/OpenAI-compat/
Anthropic/local), config-driven (YAML deep-merge + env + CLI), single/multi-turn +
optional multi-agent swarm (Generator+Critic) + bandit strategy selection, hybrid
rule+LLM judge, reproducible (config hash + seeds). **Safety posture is load-bearing:**
a `banner.assert_authorized` gate refuses to launch without a confirmed authorization
scope; API keys are read by env-var NAME and NEVER logged; the attacker has no network
egress beyond its model server; strategy modules implement PUBLISHED techniques as
mechanics/scaffolds only (no baked-in working exploit payloads — real seeds come from
the runtime goals YAML). Built for defensive discovery + reporting, mirroring PyRIT/
Garak/DeepTeam. Phases 2-6 (multi-turn, TAP, deeper agentic, reporting) layer on the
same interfaces.

**Operational playbook (hard-won on this host — follow these):**

1. **Build a lesson with a ~5-agent team on a shared interface contract.** One
   agent per file-group (core / train / eval / README / app), each CPU-only
   ("write + import-check, do NOT load the model"), disjoint scopes. The lead
   (you) defines the exact function signatures + the `results.json` schema in
   every prompt, relays signatures between blocked agents via `SendMessage`,
   import-checks the whole package, reconciles interface drift, and does ALL
   commits centrally (agents never run git). Reuse lesson-2
   `model_utils`/`judge`/`gate` and lesson-1 `probe.py` rather than re-writing.
2. **Host RAM is the real bottleneck, not VRAM.** Chrome routinely holds ~28 GB
   of the 32 GB; when free RAM < ~4 GB a bf16 Gemma-1B load OOM-dies and
   generation pages to disk (~36 s/gen vs ~2 s). Check free RAM before a GPU run;
   if low, ask the user to close tabs. **Background GPU jobs get reaped under RAM
   pressure — run model jobs in the FOREGROUND**, or background only when RAM is
   healthy. Every `run_*.py` takes an env cap (e.g. `REFT_EVAL_N`,
   `REALIGN_N_EVAL`/`REALIGN_ALPHAS`) to shrink an eval into one foreground
   window; results are screening-tier and labelled as such.
3. **Windows cp1252 console kills unicode.** Never print `α`, `Δ`, `‖`, etc. to
   stdout in a runnable script — the summary print crashes with
   `UnicodeEncodeError` (use `alpha`/`Delta`/`||`). `results.json` + plots are
   always saved BEFORE the summary print, so a late crash still leaves the data.
4. **Load gated Gemma from the LOCAL path** `models/google/gemma-3-1b-it` (the HF
   id 401s without a token); the abliterated model is `DavidAU/…-heretic-…`.
5. **Training the stiff two-term loss** (refusal-CE + benign-KL) oscillates —
   always use gradient clipping + best-checkpointing (save the lowest-loss step,
   not the last). A single unconditional vector cannot both refuse-harmful and
   spare-benign; that tension is why steering is applied through the gate.
6. **Report honestly:** benign over-refusal here is dominated by the base
   abliterated model + the weak 1B self-judge (the instrument), not the method —
   say so. `n` is screening (§7); never call a screening result a "win".
7. Latest-developments requests → `WebSearch` and filter by the arXiv `YYMM`
   prefix (`26MM` = 2026); do not cite from training memory without a date.

---

# 18. SESSION STATE — portable checkpoint (maintained; last update 2026-07-27)

> Written so the session can be resumed on **another machine**. Read §18 before §16.
> Three repos are active and all are public under `github.com/dlmastery/`.

## 18.1 Repos and their state

| repo | purpose | state |
|---|---|---|
| `steeringresearch` (this) | activation-steering autoresearch + `steering_tutorials/` course | loop RESTARTED; first legitimate champion (HC-1) |
| `auto-research-voice-based-disease-detection` | voice-health **claim-audit** program | apparatus built; **0 experiments run** |
| `yoganext` | agent-first meditation app | complete; 24 tools, 25/25 UI parity, verified failing-then-passing |

## 18.2 What was established this session (all pushed)

**The forensic critique of the old loop.** 124 experiments, 0 external-ready findings.
Five root causes, each evidenced: (1) champion `exp3` was a Qwen-0.5B row with *stubbed
safety* and a self-admittedly *circular* behavior proxy, frozen 2026-05-30 across 121
later experiments; (2) the composite fingerprint (`a9001e87087e`) no longer matches the
code (`8509c229b58f`) and 92/124 rows do not reproduce; two of five priced axes were
inert; (3) the 7-step ritual was decorative — `_manual: true` on all 124, 81 identical
citation pastes, **0/124 falsifiable numeric predictions**; (4) 94 % synthetic substrate,
74 % on a 270M model its own findings called unsteerable; (5) `n≥7 + Holm` is
**arithmetically unsatisfiable** for m ≥ 4 — which is why the contract was met **zero**
times. Fixed in `stats.py` (`power_note(..., family_size)`) with a regression test.

**The instrument is broken and the program is re-scoped around it.** Two judge
calibrations failed: AUC **0.665** (integer readout) → **0.7508** (continuous
expected-value readout, +0.086 and 12.8× faster). Both under the 0.85 gate. See
`autoresearch_results/JUDGE_CARD.md`. **No judge-dependent claim is admissible.**
All live work uses judge-free endpoints: WikiText-2 perplexity, geometry, probe AUC
vs ground-truth labels.

**Literature refresh (Jun–Jul 2026), 4 scans in `corpus/LIT_2026-07_*.md`.**
**CORRECTED 2026-08-08 — all eight IDs were WebFetch-verified for the first time in
`corpus/LIT_2026-08_conformal_gate_novelty.md`; the original claim below was wrong in
three ways. The scoop is TWO papers, not seven, and both were already known here.**

- **Genuine scoops (2):** `2606.06735` (angle-norm decomposition — independently reached
  HC-1's own angle/radius parameterisation, 7 models to our 1) and `2602.06801`
  (non-identifiability). Both were **already known to the program** — `2602.06801` has a
  `non_identifiability` lesson built against it with an unrun alpha sweep (§18.6 item 3).
  The accurate account is **not** "scooped while idle" but **"built lessons against
  published results and did not run them."**
- **`2606.20852` — STRUCK. FALSE CITATION.** It is *"Activation Steering for Pneumonia
  Classification on Chest X-rays"* (Farina et al., 18 Jun 2026, cs.CV). A radiology paper.
  Its only link to direction fungibility is the phrase "activation steering." It survived
  because it *resolves* — the §18.8 pattern, failing silently and plausibly.
- **`2601.19375` — RE-FILED AS AN ADVERSARY, NOT A SCOOP.** *Selective Steering:
  Norm-Preserving Control…* (Dang & Ngo, 27 Jan 2026) pitches **norm preservation as the
  GOOD regime** and blames its violation for collapse "particularly in models below 7B."
  **HC-1 measured the opposite in that same band** (r=0 worst; pure rotation 4× the PPL of
  pure addition at f=0.10). HC-1 does not lose to this paper — it **contradicts** it, which
  is sharper than the contribution §18.2 mourned. Filing an adversary as a scoop is how a
  live disagreement gets silently retired. Re-run HC-1 to that paper's protocol before
  claiming it. Its layer-selection half also belongs in **HC-3**'s citations.
- **Overstated (2):** `2604.09839` proves off-manifold-ness (a binary formal property), not
  a displacement budget. `2509.22067` is the §10 Rogue Scalpel paper, **v1 26 Sep 2025** —
  a design input since §10 was written, not a Jun–Jul 2026 discovery.
- **The conformal gate survives, and the sweep is now evidenced**: exhaustive arXiv
  full-text conjunctions return **zero** on-topic hits; `"conformal"` + `"residual stream"`
  returns **zero papers in all of arXiv**. `2603.14623`'s AUC-feasibility reading is
  CONFIRMED at theorem level (Prop 2, Thms 1–2) — **but its Remark 2 shows AUC is
  sufficient-not-necessary and it routes successfully at AUC 0.57.** So a better gate is
  *not* what it is waiting on: our 0.7508 judge is already inside its working regime, the
  0.85 gate is **self-imposed**, and the real opening is that **feasibility is a LOCAL ROC
  property while we report global AUC everywhere**. Time-limited: `2605.14746` (the Romano
  conformal group) names the exact problem one layer away, without conformal.

**Findings that survive:**
- **HC-1 (new champion).** Angle/radius at fixed chord is **monotone**, not U-shaped:
  norm preservation (r=0) is the WORST allocation; at f=0.10 pure rotation costs 4× the
  perplexity of pure addition. Champion `f=0.05, r=1.0`, PPL 87.652 = 0.949× base.
  Challenges the mechanism GEMS/ORBIT assume for stacking.
- **Matched-budget.** Rotation worse than addition at 5/5 budgets, judge-free.
- **flas v2.** Mis-scaled dial fixed (norm-relative); the corrected sweep is a *clean*
  negative — at T=0.02 coherence is intact yet refusal already falls 0.32→0.26.
- **Voice F1.** On SVD, **age alone → ROC-AUC 0.871**; 200/1853 speakers have >1 session.

## 18.9 SESSION CHECKPOINT — 2026-08-17 (NEWEST — read before 18.6)

### Two instrument bugs, each of which silently manufactured a "finding"

**1. EmbeddingGemma has been running CAUSAL, not bidirectional.**
`transformers` 4.55.0 contains **zero** references to `use_bidirectional_attention`,
so EmbeddingGemma-300M's config flag bound to nothing and was silently dropped.
Proved by BEHAVIOUR, not by reading the config: two sequences sharing a 9-token prefix
gave **bit-identical** hidden states (`max |Δh| = 0.000000e+00`). After upgrading to
**transformers 5.15.0** the same test gives **5.74**. Test shipped as
`audits/test_embedder_bidirectional.py`; full writeup in
`audits/AUDIT_2026-08-17_embeddinggemma_causal.md`.

- **`biencoder_guard` results SUSPENDED** (not reversed). Its conclusion — bi-encoder
  loses to a trained head (macro-AP 0.240 frozen / 0.575 contrastive vs 0.658), binary
  harm AUC ~0.59 against a 0.526 length confound — has a crippled encoder as a
  *sufficient* alternative explanation. **What survives** (attention-mode-independent):
  the **43.9×** latency/scaling result, and that a trained head structurally cannot
  score a held-out policy.
- **`cross_trajectory` gemma ablation — I SUSPENDED IT, THEN WITHDREW THAT.** I read
  `"embedder": "gemma"` as EmbeddingGemma. `config.py:70` defines
  `EMBEDDER_CHOICES = ("embeddinggemma", "gemma", "minilm")` as **three distinct
  options**: `"gemma"` is a Gemma-3-1B **decoder** layer-12 residual (`hidden: 1152`),
  causal *by design*, so a dropped bidirectional flag cannot touch it. Suspension
  withdrawn; file restored to `results_gemma_ablation.json`. Its real defect is the
  pre-existing one: **hand-transcribed from a run log, not regenerable from the code.**
  *(Caught by a subagent reviewing my own change. Blast-radius estimates need the same
  verify-the-behaviour discipline as the bug that prompted them — I matched on a
  substring.)*
- Unaffected: `meerkat` (bge/minilm), `cross_trajectory/results.json` (minilm),
  `cross_trajectory` gemma-decoder arm. `cross_trajectory`'s actual `embeddinggemma`
  arm is `[PENDING RUN]` — no number exists to suspend.
  `multiturn_jailbreak` was `PENDING_RUN` with no metrics — nothing to suspend.
- The upgrade to 5.x is a MAJOR bump; it was **verified, not assumed**: flas self-test,
  a real Gemma-3-1B forward + hook (`hook restores exactly`), imports across 5 lessons.
- **All causal `.npz` caches quarantined** to `_causal_invalid_caches/` (gitignored).
- **OPEN:** cache keys fingerprint the DATA, not the encoder's BEHAVIOUR, so a naive
  re-run would silently reuse causal vectors. Deferred *on purpose* while the SHADE run
  was in flight; do it after. Version alone is not a sufficient key — a config field and
  a version string both said "bidirectional" while the behaviour said otherwise.

**2. `flas` v2 did not measure a dial; it diverged to NaN.** Field trained on
last-token activations only while `FlowContext` transports every position ⇒ `‖v‖`=11.7
off-distribution, and the step `dt·‖x‖·v` is a feedback loop: rel displacement
0.69 → 6.09 → **99694**, NaN at ~10× residual norm ⇒ empty completions ⇒ **graded
GIBBERISH and published as "the citable one."** Fixed (all-position training,
`unit_velocity` making displacement `T·‖h‖` by construction, `BROKEN` verdict,
representative probe). Guard verified load-bearing: **0/16 cells fail with the fix,
16/16 without**. v2 artifacts quarantined; README retracted in place. **Retrain +
re-run still PENDING.**

### STA (`streaming_trajectory_aggregation`) — the live experiment
Apparatus complete. **`results_agentdojo.json` is absent because the run was
DELIBERATELY DISCARDED, not because persistence failed** — traced 2026-08-17. The
write path is sound (`json.dump` → `[write]` → summary, `run_sta.py:431-434`) and the
log contains its `[write]` line at line 37, so the file was created and then removed
by commit `581dd98` ("the in-flight run is INVALID and discarded") after a code audit
found three critical bugs. `_sta_agentdojo.log` is a leftover of that discarded run.
**Do not re-derive a persistence bug from its absence.** The log is still not evidence
(§18.8) — but because the run was voided, not because the harness lost it. That log
recorded
F0/F2/F3 **HOLD**, **F4 FAILS** (esn_cusum 0.796 vs binding bar 0.889; safetydrift
0.386) — and it ran on causal embeddings, so F4's failure is doubly suspect.
**SHADE run launched 2026-08-17** (n=500/class, pool 749/744, median 131 steps,
`has_lead_time=True`) — the only corpus that can evaluate **F1** (the horizon
hypothesis) and **F5** (lead time); both were `N/A` on agentdojo.

### Literature added
`corpus/LIT_2026-08_longcontext_crosstrajectory_latent.md` — 16 papers, 9
WebFetch-verified. Unifying frame: **every attack is the same move at a different
granularity — fragment the signal below the defender's scoring window**. Two findings
that bear on existing lessons: SALO (2605.02958) says mean pooling *dilutes* the sparse
refusal needle (our `hello_world` mean-pools at a fixed layer 12; `probe_tuning`'s layer
sweep still has no code); and 2604.28129 attributes **50–59% FPR to binary turn labels**
(`trajguard`/`multiturn_jailbreak` both label binary). Gate on everything latent:
**2412.09565 drops latent-defence recall 100%→0% at ~90% jailbreak rate**, and every
2026 method above is measured against non-adaptive attacks only.

### The rule these two bugs pay for
**A library warning about a silently-ignored parameter is a BLOCKER, not noise** — and
**verify the behaviour, never the config field.** The config said `True` the whole time.

## 18.6 SESSION CHECKPOINT — 2026-08-02 (read this first on a new machine)

### What this session did, in one line
Audited the 22-lesson course against the papers it cites, found **6 lessons
misattributing results**, fixed them, and hill-climbed the research program onto a new
champion. **Most of the value is corrections, not new claims.**

### Landed and pushed (all on `master`)

**Research program**
- **HC-3**: champion moved **layer 12 -> 11** (paired -1.812 PPL, CI [-2.383,-1.346], 8/8
  seeds). EVALUATION-eligible, **NOT external-ready** — the ordinal gate FAILS. The
  norm-matched random control at every layer is what made it a win: L17 looked competitive
  on PPL (73.9 vs 73.3) but its random control was also nearly free, so its low perplexity
  was layer INSENSITIVITY. L23 is a true null, L25 is worse than random.
- **HC-S**: `diffmean` DEFENDED on the source axis (pca +3.334, lda +24.202, random
  +46.900, all CIs excluding zero, 8/8). Key: **cos(diffmean,pca)=0.966 is above the usual
  0.95 "same direction" bar and STILL costs +3.33 PPL** — a 0.95 cosine bar is NOT
  sufficient to call two steering directions interchangeable.
- **M-b**: took THREE bases to get one answer. `gs` (displacement fixed, directions NOT
  exchangeable) says the gap SHRINKS; `raw` (exchangeable, displacement triples) says it
  GROWS; only `rawnorm` (both controlled) is right: **FLAT**, +88.89/+86.59/+85.95/+85.92
  at N=1/2/4/8, additive winning 32/32 cells.
- **best_config.json**: the discredited Qwen-0.5B champion was still in it and is now
  quarantined to `best_config_LEGACY_QUARANTINED.json`.

**Course corrections (6 lessons were misattributing results)**
- `biencoder_guard`: the bi-encoder arm was FROZEN cosine while its papers FINE-TUNE.
  Added `ContrastiveBiEncoderGuard` (InfoNCE over frozen backbone). Then the **projection
  init was wrong too** — `torch.eye` truncation zeroes 512/768 dims; random orthonormal is
  now default (measured: 3.7x better absolute-cosine preservation). Added **EXP-G**
  (~900 unrelated distractors) and **EXP-H** (Opir's 996-category 3-level taxonomy).
- `hello_world_steering`: extracted its refusal direction from the ABLITERATED model (no
  refusal to difference). Fixed to extract from the aligned base. Alpha grid extended to
  0.25. **Refusal rises NOWHERE** — the vector removes compliance and destroys coherence.
- `realignment`: **headline FALSIFIED.** ASR 0.46->0.045 is **99% incoherence, not
  refusal** — at alpha=0.25, of 191 non-jailbroken outputs only **2 are REFUSAL, 189
  GIBBERISH**. Its "coherence" was distinct-token ratio, which REWARDS the failure
  (whitespace collapses, one huge token is trivially 100% distinct).
- Plus doc corrections in `multi_intent`, `reft_r1`, `fine_grained`, `contextual_steering`,
  `decomposing_prompting`, `trajguard`, `meerkat`, `cross_trajectory`, `stacking`,
  `curveball`, `flas`, `prompt_activation_duality`, `probe_tuning`, `talan`.

**Judge provenance (a class bug)**
`Judge` silently self-judged when `STEER_JUDGE_MODEL` was unset, making R17.3
unenforceable by inspection. Now `Judge.stamp()` records
`{judge_id,is_self_judge,judge_model_id,off_family}`; 16 runners stamp it plus seed.
Inventory: **7 verified off-family**, **1 confirmed self-judged (`gavel`)**, **8 UNKNOWN**
(decomposing_prompting, hello_world_steering, multi_intent, non_identifiability,
prompt_activation_duality, reft_r1, rogue_scalpel, stacking/run_stacking). *Absence of
evidence is the defect, not a clean bill.*

### PENDING — what to run next on the new machine

All of these are BLOCKED ONLY BY HOST MEMORY on the old box, not by missing code:

1. `gavel/rejudge.py` — code is written, CPU-verified, 3-phase, one model resident at a
   time, hard anchor assertion. Run the four commands in gavel README "Run it".
   Only the LEAKAGE row is inadmissible; block rates and per-CE firing are judge-free.
2. `realignment` benign half — only 25/200 measured; `over_refusal` is carried from the
   prior run and labelled as such.
3. Unrun alpha sweeps — **CORRECTED 2026-08-17, this entry was STALE in two ways**:
   - `fine_grained` — **its sweep HAS run.** `artifacts/results.json` carries a populated
     `sweep` key plus `dense` / `best_sparse` / `thresholds`. Remove it from this list.
   - `non_identifiability` — **genuinely unrun.** No `sweep`/`alpha` key; only a single
     `matched_alpha`. Still open.
   - `rogue_scalpel` — **its guard LADDER has run** at n=200 (baseline / attacked /
     +clamp / +lock / +dual). What is actually open is (a) the `negative_add` attack
     mode — the artifact records `attack_mode: "project_out"` — and (b) a *per-layer*
     A–E ablation; the shipped ladder covers A(lock)/B(clamp)/D(dual) at one layer only.
     Its README already reports the ladder honestly, including that `+lock`/`+dual`
     collapse to gibberish 1.00 and are WORSE than `+clamp` alone — a correctly
     reverted rung, per §9.
   *(Generalise: this list was trusted for weeks without checking the artifacts beside
   it. Verify contents, never a remembered status.)*
4. `meerkat` — `bge` is the config default AND the paper's own encoder, never run;
   all numbers are MiniLM.
5. `probe_tuning` — the promised layer sweep has NO code and NO artifact (marked KNOWN GAP).
6. `hello_world` 4B conditional arm — crashes on dim mismatch (1152 vs 2560).
7. Re-run the 8 UNKNOWN-judge lessons under the new stamping to establish provenance.
8. Research: budget_f axis (f<0.02, f>0.10 unexplored), span axis, HC-3's open L17/L20
   cells, and a quiet-GPU backbone re-measure for the withdrawn scaling claim.

### Host traps that cost the most time here
- Windows **COMMIT LIMIT**, not free RAM, is usually binding. `OSError 1455` = commit
  exhausted. A **silent exit with no traceback** = physical memory gone mid-mmap of judge
  shards. Gate on BOTH (commit >= 8.5 GB, physical >= 7.5 GB).
- The harness **REAPS** long background jobs. Checkpoint at the granularity of the most
  expensive irreversible step, not what is convenient to write.
- Latency ratios formed across two machine states are meaningless — see the withdrawn
  64x / 40x / 43.91x sequence. Quote only a run with `contended:false` in BOTH witnesses.

## 18.3 KNOWN DEFICIENCIES — status as of 2026-07-28

| # | deficiency | status |
|---|---|---|
| 1 | **Voice program had run ZERO experiments** (the `darebench` failure mode: many scaffolds, no research) | **IN PROGRESS.** V2 is the first experiment ever run against a registered hypothesis. Confirmatory n=10 in flight. 6 of 7 still UNTESTED. |
| 2 | Voice dashboard was one flat page | **DONE.** `docs/hypotheses/` adds the per-hypothesis tier (V1–V7), generated from `IDEA_TABLE.md`. Renders the PREDICTION whether or not a result exists, so an untested hypothesis is a visible debt. Tier chips read the run's own repeat count — a 1-repeat smoke file must not look like a 10-repeat confirmatory file. Build-time markdown-leak gate. |
| 3 | Voice has 2 audits; steering has 17 | **OPEN.** Missing: impl-critic, sci-critic, **data-split audit** (the one this program is about), shuffle-test, meta-process. |
| 4 | `trajguard` claimed AUC 0.944/0.945 with NO confound baseline | **DONE.** Bar = **COMPLETION** char-length **0.7354** directionless (prompt char-length is at chance, 0.5032 — you cannot call this from the prompt alone). Margins: `seq_gru` +0.209, `trajectory_mlp` +0.208, `per_turn_max` +0.195, and the paper's own `threshold_freeform` (0.638) lands **below the bar**. |
| 5 | `meerkat` never stated its length discount | **DONE, and it went further.** `results.json` predated the `data.py` length-matching fix and could not be regenerated from the code beside it. Re-run at identical config: per-trace −11% (0.818→0.731), **clustering arm −65% (0.568→0.199)**. `pool_fingerprint` guard added; repo-wide staleness sweep found the defect isolated to this one lesson. |
| 6 | HC-1's direction from only 10 harmful / 8 harmless prompts | **DONE** — superseded by M-a (340 harmful / 120 harmless real benchmark prompts, n=8). |

## 18.7 VOICE — three registered hypotheses executed (2026-07-28)

The program went from **0 experiments** to **3 of 7 hypotheses with results** and
**6 findings**. All judge-free (ROC-AUC vs clinical labels), so the failed judge
calibration is irrelevant to every number below.

### F4 / V2 — EVALUATION tier. ~a third of WavLM's discrimination is speaker IDENTITY

n=10 x 5-fold speaker-disjoint, m=14 pre-registered, **4h15m CPU**. Audits
arXiv:2604.14354, which established speaker leakage by *measurement*; the mechanistic
removal had not been run.

| k | AUC spk-removed | AUC topk-removed | D vs topk [95% CI] | var spk/topk |
|---|---|---|---|---|
| 1 | 0.6755 | 0.7320 | +0.0565 [.056,.058] | 0.206/0.345 |
| 8 | **0.6104** | 0.7025 | **+0.0921** [.090,.094] | 0.700/0.812 |
| 64 | 0.5398 | 0.6287 | +0.0888 [.086,.091] | 0.940/0.959 |

Full AUC 0.7382. D positive with CI excluding zero at **all 7 ranks, both controls**.
Cleanest form: at *every* rank the speaker subspace removes **less** variance than
top-k PCA yet costs **more** AUC. Identity = **24–39%** of above-chance headroom.
**Shuffle control passes**: D 0.0921 → 0.0016, full AUC 0.5074.

**The falsifier did NOT fire.** It required D's CI excluding zero *and* AUC_spk
including 0.5; at k=64 AUC_spk is 0.5398 — near chance, not chance. So
"clinical validity falsified" **does not follow**. Manipulation check remains WEAK
(speaker-ID 0.278 → 0.193 vs a predicted 0.90 → <0.30) — mean-pooled WavLM-base+ is
not an x-vector.

### F5 / V6 — PARTIAL. Scaler-before-split leakage is *nothing* on SVD embeddings

`D = +0.00004` [+0.00001, +0.00007] on WavLM; **−0.00418** on eGeMAPS with a CI
excluding zero (leakage *degrades*, reproducing the audited paper's subtler point).
Extends the published handcrafted-feature near-null to **embeddings**. The
corpus-specificity claim — V6's actual novel content — is **not evaluable**.

### F6 / V7 — PARTIAL. The Clever-Hans silence shortcut does NOT generalise

SVD 0.5136 · Coswara 0.5146 (directionless) · COUGHVID 0.5264. 42,654 files VAD'd,
0 unreadable, nothing above 0.527. **My predicted mechanism was WRONG**: I predicted
Coswara ∈ [0.55,0.70] because it is crowd-recorded and heterogeneous; it came in
**lowest**. Caveat that limits the null: these corpora are sustained vowels/coughs —
Pitt is spontaneous speech where pause structure carries cognitive load, so this is
**not** evidence the Pitt effect was spurious. PROCESS-2 is the sharper test.

### COUGHVID extraction — DELIBERATELY ABANDONED, with cause

Killed twice at ~8.5% of 13,535 files (backgrounded GPU jobs get reaped on this host;
the extractor writes one npz at the end, so each retry loses everything). Not retried,
because **F2 already established COUGHVID can never carry an evaluation claim** — it
has 13,535 unique ids for 13,535 recordings, so its `GroupKFold` is plain `KFold` in
costume. Spending hours of GPU to populate cells whose results are unclaimable, in a
program whose thesis is that unclaimable numbers get reported as claims, is a bad
trade. Recorded as a **deliberate exclusion citing F2**, not a silent omission.
*A pre-registered cell that becomes known-uninformative is different from one that is
merely inconvenient, and the distinction belongs in the record.*

## 18.4 Next actions, in order

1. Finish V6 re-run (SVD + Coswara = 3/6 cells) and update F5.
2. **V1 then V3** — both now unblocked for SVD + Coswara by the Coswara extraction.
3. V4 (free once V1/V3 predictions exist), V5 last (GPU-gated).
4. Voice: the shuffle-test and meta-process auditors are still missing.
5. Steering: `cross_trajectory` Gemma-embedder ablation; N-vector stacking vs GEMS/ORBIT.

## 18.8 THE RECURRING FAILURE MODE — read this before debugging anything

Every serious defect found in this program failed **silently and plausibly**, never
loudly. Not one crashed. Each produced confident, well-formed, wrong output:

- `load_rows()` returned 0 and the dashboard **silently dropped every v2 row**.
- `preprocess_audio.py` globbed the 22-zip PARTIAL while reporting `complete: True`,
  capping the corpus at 49 of 1,679 speakers.
- The embedding cache returned **stale labels** under a key that ignored them.
- `meerkat`'s `results.json` **could not be regenerated from the code beside it**, and
  the README priced every method against a confound bar that no longer existed.
- A `str.replace` with a stray trailing space **matched nothing and returned
  silently**, so a "successful" build shipped a page with literal markdown in it.

The operative rules, each paid for: **assert your anchors** (a replace that matches
nothing must fail, not pass); **stamp your inputs** (an artifact that cannot be
regenerated from the code beside it is not evidence); **gate at build time** (a check
that must be remembered will eventually be skipped); and **verify contents, never
listings** (a directory listing said the model was cached; it was a 16 KB stub).

## 18.5 Host gotchas (cost real time — do not repeat)

- **Use `C:\Users\evija\anaconda3\python.exe`.** Bare `python` is Windows-Store 3.13:
  no CUDA, `transformers` will not import.
- Windows **cp1252** console: printing a German filename (umlaut) crashed a 38 GB
  download. Use `PYTHONIOENCODING=utf-8` and `.encode('ascii','replace')`.
- Piping a long job through `head -n` **SIGPIPE-kills** it.
- **Harness-tracked background jobs get REAPED; orphaned ones do not.** Three kills this
  session with RAM healthy each time (COUGHVID WavLM at 1,152/13,535; SVD eGeMAPS at
  21,200/28,509). Meanwhile the V2 run survived 4h15m -- but only because a stray `&`
  had orphaned it from the harness. So the fix is NOT more headroom or shorter jobs:
  **make long jobs resumable.** `extract_egemaps_resumable.py` checkpoints every 500
  files and skips completed paths on restart, so a reap costs a minute instead of hours.
  Compute deserves the same treatment as bandwidth (`fetch_svd_resumable.py`).
- **Never put `&` inside a `run_in_background` call.** Any form of it — `nohup … &`,
  `cmd > log 2>&1 &`, `cmd & sleep 5` — orphans the real job: the *wrapper* exits
  immediately, the completion notification fires for the wrapper, and the actual
  process keeps running with nothing watching it. Hit twice now (SVD download; the V2
  n=10 run). Pass the command directly and let `run_in_background` own it.
- Redirecting a long Python job to a file gives an **empty log until it exits** —
  stdout is block-buffered when not a tty. Use `python -u` if you want to tail it.
- **ONE GPU.** Re-check `nvidia-smi --query-compute-apps` before every launch — three
  concurrent model loads got two jobs killed this session.
- `google/embeddinggemma-300m` is **gated** on the hub and this host has **no HF token**
  (`huggingface-cli login` required). Metadata listing succeeds while weights 401 — test a
  real weight fetch, never a file listing.
  **UPDATED 2026-08-08: the MODEL blocker is STALE — the EmbeddingGemma weights are already
  on disk** (found by the `multiturn_jailbreak` audit). So the §17 encoder mandate is
  *unblocked* and MiniLM/BGE headlines are non-compliance, not necessity. **The DATASET
  gating is real and unchanged**: verified `gated` and unusable here are `ScaleAI/mhj`
  (which §17 rule 5 names as this family's benchmark — its local cache holds only a 40-byte
  `refs/main`), `allenai/wildjailbreak`, `allenai/wildguardmix` (403), `walledai/HarmBench`,
  `lmsys/lmsys-chat-1m`. Verified **ungated** and usable today:
  `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` (33,416 rows, 3-level taxonomy),
  `SafeMTData/SafeMTData` config `SafeMTData_1K` (1,680), `tom-gibbs/multi-turn_jailbreak_attack_datasets`
  (1,200 purpose-built Semi-Benign hard negatives), `intrinsec-ai/cstm-bench` (already cached).
  *Verify contents, never listings — in both directions: a listing said the model was gated,
  and the weights were sitting on disk.*
