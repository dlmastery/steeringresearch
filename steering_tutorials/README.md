# steering_tutorials — a progressive course in activation steering

A hands-on, standalone tutorial series that teaches **activation steering of
LLMs** from scratch, on small Gemma models you can run on one GPU. Each lesson
is a self-contained package (code + rigor + a demo webapp), independent of the
research harness in `src/steering`. Lessons add **exactly one new idea** at a
time and climb in difficulty.

The spine is a three-verb arc that then branches into control, safety, and rigor:

> **READ** (detect a concept in activations) → **WRITE** (push along a direction)
> → **GENERATE** (learn to produce directions) → **CONTROL** (when/how-far/how)
> → **CERTIFY** (guarantees) → **DEFEND** (adversarial) → **PROVE** (rigor & scale).

Everything here is grounded in this repo's research (the `hypotheses/`, `corpus/`,
and `src/steering/` work) — the lessons are that research, repackaged for
learning.

---

## Course map

| # | Lesson (dir) | Stage | Teaches | Compute | Status |
|---|---|---|---|---|---|
| 1 | [`hello_world`](hello_world/README.md) | READ | linear/shallow probing of activations for a concept (harm) | light | ✅ built + validated |
| — | [`probe_tuning`](probe_tuning/README.md) | READ+ | layer sweep + MLP hyperparameter search (CV, no test peeking) | light | ✅ built + validated |
| 2 | [`hello_world_steering`](hello_world_steering/README.md) | WRITE | CAA/diff-of-means steering vector, conditional gating, an LLM judge | med | ✅ built + validated |
| 3 | [`reft_r1`](reft_r1/README.md) | GENERATE | AxBench's learned rank-1 ReFT intervention; ReFT-r1 vs DiffMean vs prompting bake-off | med | ✅ built + validated |
| 3b | [`flas`](flas/README.md) | GENERATE+ | flow-based steering: a concept-conditioned velocity field; flow-time = strength dial | med | ✅ built + validated |
| — | [`non_identifiability`](non_identifiability/README.md) | 2026 | steering vectors aren't unique — many low-cosine directions, same effect (arXiv:2602.06801) | med | ✅ built + validated |
| — | [`fine_grained`](fine_grained/README.md) | 2026 | sparse (top-k%) edits vs dense — "steering less" (inspired by AUSteer, arXiv:2602.04428) | med | ✅ built + validated |
| — | [`contextual_steering`](contextual_steering/README.md) | 2026 | per-input adaptive steering strength (inspired by CLAS, arXiv:2604.24693) | med | ✅ built + validated |
| 4 | `displacement_budget` | CONTROL | the coherence cliff; bound off-manifold displacement | light | planned |
| 5 | `operations` | CONTROL | add vs project-out (ablation) vs rotate (norm-preserving) | light | planned |
| 6 | `fungibility_null` | CONTROL | direction controls: shuffled/random/orthogonal — is the vector special? | light | planned |
| 7 | `conformal_gate` | CERTIFY | conformal prediction → a provable benign over-refusal bound | light | planned |
| 8 | `sae_gate` | CERTIFY | interpretable SAE-feature gates with human-readable firing reasons | med | planned |
| 9 | [`multi_intent`](multi_intent/README.md) | CONTROL | steer K concepts at once; orthogonalization; the norm budget | med | ✅ built + validated |
| 10 | [`rogue_scalpel`](rogue_scalpel/README.md) | DEFEND | red-team the guard: the universal attack + the five-layer defense | med | ✅ built + validated |
| 11 | [`realignment`](realignment/README.md) | DEFEND | restore refusal in an abliterated model by transplanting a direction | med | ✅ built + validated |
| 12 | [`stacking`](stacking/README.md) | PROVE | which priors stack (orthogonal sites) vs compete (same site) | med | ✅ built + validated |
| 13 | `composite_metric` | PROVE | the Goodhart-resistant multi-objective score; Pareto fronts | light | planned |
| 14 | `stat_rigor` | PROVE | screening vs evaluation; Wilcoxon + bootstrap + Holm-Bonferroni; HARKing | light | planned |
| 15 | `scale_fungibility` | PROVE | does the direction start to matter at 9B? endogenous steering resistance | heavy (A100) | planned |
| 16 | `certified_deployment` | CAPSTONE | certified gate vs a classifier-router under attack, on ASR/over-refusal/MMLU | heavy | planned |

Prereqs cascade: each lesson assumes the ones before it, but every package runs
on its own (it re-derives or imports what it needs from earlier lessons).

---

## Rigor & honesty (the measurement stack)

Every steering lesson is validated the same, honest way:

- **Data:** a shared **≥500 harmful + ≥500 benign** set (`common/data.py`) — 100%
  lmsys/toxic-chat, deduped, **length-matched (length-AUC 0.501** — the classes
  carry *zero* length signal, so a probe/vector can't cheat on length).
- **Judge:** an **off-family Qwen2.5-3B** grades REFUSAL/COMPLIANCE/GIBBERISH
  (`STEER_JUDGE_MODEL`). The tutorials' tiny **1B self-judge inflates refusal** —
  it mislabels softened/hedged compliance as REFUSAL — so results judged by it are
  *not* trustworthy. All headline numbers below use the Qwen judge.
- **Papers:** an independent auditor per lesson (`AUDIT.md`) WebFetch-**verified
  every cited arXiv ID as real** and fixed attribution errors.
- Numbers are **screening-tier** (`n ≈ 50/class`), labelled as such.

## Status & validated results (2026-07-16, honest re-run)

The honest through-line the re-runs revealed: **learned interventions survive a
real judge on hard data; simple fixed diff-of-means vectors largely don't** — and
the earlier rosy numbers were inflated by the 1B self-judge + easy in-distribution
JailbreakBench. Reported as they landed, negatives included.

| Lesson | Stage | Honest result (Qwen judge, 500/class toxic-chat) |
|---|---|---|
| [`hello_world`](hello_world/README.md) | READ | probe 5-fold CV **0.87 ± 0.03**; leakage clean; XSTest OOD AUC 0.89 (a *probe*, no judge involved — unaffected) |
| [`probe_tuning`](probe_tuning/README.md) | READ+ | 23-config sweep → simple default wins (no config beats the CV noise band) |
| [`hello_world_steering`](hello_world_steering/README.md) | WRITE | **fixed steering barely works**: refusal *falls* 0.26→0.10 as α rises, gibberish 0.20→**0.74**; JBB gate transfers poorly (**0.68**). Old 0.70/0.975 was self-judge + JBB inflation. |
| [`reft_r1`](reft_r1/README.md) (AxBench) | GENERATE | **reproduces AxBench**: learned **ReFT-r1 0.54 > DiffMean 0.26 > prompting 0.18** (steering); DiffMean wins **detection** (AUC 0.71 vs 0.61) |
| [`flas`](flas/README.md) | GENERATE+ | **payoffs don't hold** under an honest judge: zero-shot 0.67→**0.25**; higher T adds gibberish, not refusal |
| [`non_identifiability`](non_identifiability/README.md) | 2026 | **not supported at this α** — the effect is too weak for a "family" to form (setup sound; needs a stronger α sweep) |
| [`fine_grained`](fine_grained/README.md) | 2026 | **"steering less" holds** (2%-sparse as coherent as dense, gibberish 0.00) but **"achieving more" doesn't** (refusal 0.00 — the fixed vector it thins is weak) |
| [`contextual_steering`](contextual_steering/README.md) | 2026 | **partial win**: per-input α cuts benign over-refusal **0.44→0.32** (the CLAS idea), though on an abliterated model it "breaks less" rather than "refuses more" |
| [`multi_intent`](multi_intent/README.md) (L9) | CONTROL | **fully deflates**: raw == ortho == **0.00** refusal, gibberish ~1.0 — the honest judge no longer credits degenerate output |
| [`rogue_scalpel`](rogue_scalpel/README.md) (L10) | DEFEND | attack strips refusal **0.52→0.00**; the **norm-clamp guard recovers it (0.60)**; lock/dual guards don't help |
| [`realignment`](realignment/README.md) (L11) | DEFEND | **works** — clean operating point α=0.2: **ASR 0.47→0.00**, over-refusal 0.00, coherence 0.85 |
| [`stacking`](stacking/README.md) (L12) | PROVE | **inconclusive** — disjoint-site stack doesn't cleanly beat a single prior; over-stacking raises gibberish to 0.92 |

Each row links to its lesson `README.md` (method, walkthrough, `## Results —
measured vs. the claim`, caveats), its `AUDIT.md`, and `artifacts/results.json` +
plots. Repo:
<https://github.com/dlmastery/steeringresearch/tree/master/steering_tutorials>

---

## Lesson plans

### L1 · hello_world — READ  ✅
**Goal:** classify a prompt as harmful/benign from a frozen Gemma's layer-12
activations with a 3-layer MLP probe. **Teaches:** activations/residual stream,
shallow probing, the full evaluation suite (12 metrics), 5-fold CV with CIs, a
leakage/confound audit, and out-of-domain transfer. **Source:** the probing
line; `E7`/`N17` context.

### probe_tuning — READ+  ✅
**Goal:** find the best layer and MLP head by cross-validation (never test
peeking). **Teaches:** model selection discipline; why "hello world" stays
minimal. **Result:** no head beats the simple default beyond CV noise.

### L2 · hello_world_steering — WRITE  ✅ (validating)
**Goal:** make the uncensored model refuse harmful prompts by adding a
diff-of-means refusal vector, **only when the L1 probe fires** (conditional
gate), judged by the same Gemma. **Teaches:** ActAdd/CAA, relative-add, the
coherence cliff, conditional (CAST-style) gating. **Source:** `2308.10248`,
`2312.06681`, `2406.11717`, `2409.05907`.

### L3 · reft_r1 — GENERATE  ✅ (eval queued)
**Goal:** implement AxBench's **ReFT-r1** — a learned rank-1 representation edit
`h' = h + r_unit·((w·h + b) − (r_unit·h))` (train r,w,b, LLM frozen) — and
reproduce AxBench's head-to-head: **ReFT-r1 vs DiffMean (the L2 baseline) vs
prompting** on steering + concept detection. **Teaches:** representation
finetuning, learned vs fixed steering, honest baselines. **Source:** AxBench
(`2501.17148`); ReFT/LoReFT (`2404.03592`). *(Replaces an earlier hypernetwork
draft — `hypersteer/`, retired to git history.)*

### L3b · flas — GENERATE+  ⏳ next
**Goal:** flow-based activation steering — learn a concept-conditioned velocity
field `v_θ(h,t,c)` and transport activations to their steered position via flow
integration `h' = h + ∫₀ᵀ v_θ dt`, with flow-time `T` as a continuous,
zero-shot steering-strength dial. **Teaches:** flow matching / continuous
transport for steering; one checkpoint, many unseen concepts. **Source:** FLAS
(github.com/flas-ai/FLAS).

### L4 · displacement_budget — CONTROL
**Goal:** measure off-manifold displacement (Δ‖h‖, effective-rank drop,
participation ratio) and cap it to hold coherence at maximal α. **Teaches:**
*how far* you can push before text breaks — the single most useful practical
skill. **Source:** `N17` (ρ=+0.585), `E3`, `hypotheses/D_geometry/H3_displacement_budget.md`.

### L5 · operations — CONTROL
**Goal:** compare the three residual operations — `add`, `project_out`
(ablation), `rotate` (norm-preserving) — on the safety task. **Teaches:** the
OPERATION axis; why rotation hugs the activation sphere. **Source:**
`src/steering/hooks.py`.

### L6 · fungibility_null — CONTROL
**Goal:** replace the real direction with shuffled / random / orthogonal
controls under a fixed gate; is the *direction* even special at small scale?
**Teaches:** controls, nulls, honest science. **Source:** `E7`,
`hypotheses/A_foundations/H2_direction_fungible_under_gate.md`.

### L7 · conformal_gate — CERTIFY
**Goal:** calibrate the gate with split-conformal prediction so benign
over-refusal is provably ≤ α. **Teaches:** conformal prediction; why a threshold
must be calibrated (fixes the OOD miscalibration L1 exposed). **Source:**
`hypotheses/B_conditional/M8_conformal_gate.md`.

### L8 · sae_gate — CERTIFY
**Goal:** use GemmaScope SAE features as the gate; emit human-readable firing
reasons; test OOD detection. **Teaches:** sparse autoencoders, interpretable
gating. **Source:** `src/steering/sae.py`.

### L9 · multi_intent — CONTROL  ⏳ next
**Goal:** steer for K concepts simultaneously; Gram-Schmidt orthogonalize the
directions; show FPR grows sub-linearly in K and track the norm budget.
**Teaches:** compositional steering, interference, the displacement budget under
stacking. **Source:** `src/steering/multi_intent.py`.

### L10 · rogue_scalpel — DEFEND  ⏳ next
**Goal:** reproduce a universal steering attack, then implement and ablate the
five-layer guard (subspace projection lock, norm/manifold clamp, avoid fragile
mid-layers, dual-forward verdict, conditional gate). **Teaches:** adversarial
robustness of interventions. **Source:** CLAUDE.md §10, `src/steering/adversarial.py`.

### L11 · realignment — DEFEND  ⏳ next
**Goal:** transplant a base model's refusal direction into the abliterated model
and measure ASR reduction (under attack) vs coherence cost. **Teaches:**
abliteration, re-alignment, safety headroom. **Source:** `scripts/run_realign_abliterated.py`.

### L12 · stacking — PROVE  ⏳ next
**Goal:** build an additive 2→N ladder; show priors on orthogonal sites **stack**
while same-site same-direction priors **compete**; the all-on hybrid is
forbidden. **Teaches:** composition discipline, the marginal-effect ladder.
**Source:** CLAUDE.md §9, `corpus/steering-stackable-vs-competing-analysis.md`.

### L13 · composite_metric — PROVE
**Goal:** score interventions with the Goodhart-resistant 5-axis composite
(efficacy, capability, coherence, safety, selectivity); draw Pareto fronts.
**Teaches:** multi-objective evaluation; why you can't win by sacrificing an
axis. **Source:** CLAUDE.md §6.

### L14 · stat_rigor — PROVE
**Goal:** turn a result into a defensible claim — screening (n≤3) vs evaluation
(n≥7), paired Wilcoxon + bootstrap CI + Holm-Bonferroni, the EXTERNAL-READY
ordinal gate, and how HARKing sneaks in. **Teaches:** statistical honesty.
**Source:** CLAUDE.md §7.

### L15 · scale_fungibility — PROVE (needs a rented A100)
**Goal:** test whether the direction starts to matter at 9B (real beats controls)
and track the model's endogenous steering-resistance rate. **Teaches:** scaling
laws for steering. **Source:** `hypotheses/A_foundations/H4_scale_dependent_fungibility.md`.

### L16 · certified_deployment — CAPSTONE
**Goal:** put it together — the certified conditional gate vs a text
classifier-router, under attack, on (ASR-under-attack, over-refusal, MMLU).
**Teaches:** the end-to-end deployment argument. **Source:**
`hypotheses/PROGRAM_conditional_displacement.md`.

---

## How to run any lesson

From the repo root:

```bash
pip install -r steering_tutorials/hello_world/requirements.txt
python -m steering_tutorials.<lesson>.<script>   # e.g. steering_tutorials.hello_world.train_probe
```

Each lesson's own `README.md` has its concepts, code walkthrough, run commands,
and honest caveats. Standards across the course: full metric suite with CIs,
a leakage/confound audit on every dataset, out-of-domain checks, principled
dataset sampling, fixed seeds, and degradations reported as prominently as wins.

Repo: https://github.com/dlmastery/steeringresearch/tree/master/steering_tutorials
