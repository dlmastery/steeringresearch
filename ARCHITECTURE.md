# ARCHITECTURE — Steering Research Autoresearch Repo

> **Scope.** This is the *engineering map* of the repository: modules, their
> dependency direction, the numpy/torch seam, the end-to-end data flow, where
> state lives, how to add an experiment driver, and the offline-testability
> story. It **complements** `src/steering/DESIGN.md` (which is the *method spec*
> for the conditional multi-intent CAST steerer) and `CLAUDE.md` (the research
> constitution: ladder, composite, rigor floor). When DESIGN.md and this file
> overlap, DESIGN.md is authoritative for the CAST method; this file is
> authoritative for repo layout and module wiring.

---

## 1. Layered module dependency graph

Dependencies point **downward only** (upper layers import lower layers; never the
reverse). Within `src/steering/`:

```
                         ┌─────────────────────────────────────────────┐
  L4  CONSUMERS          │  scripts/*.py  drivers   │  dashboard.py      │
  (drivers + reporting)  │  (run_axbench_*, run_e*, │  (master + per-hyp │
                         │   confirm_e7*, safety)   │   + per-exp HTML)  │
                         └───────┬───────────────────────────┬──────────┘
                                 │                            │
                                 ▼                            ▼
                         ┌──────────────────────────────────────────────┐
  L3  ORCHESTRATION      │  runner.py                                    │
  (the experiment loop)  │   run_single_experiment / _log_and_checkpoint │
                         │   (writes ledger, best_config, reasoning,     │
                         │    triggers dashboard build)                  │
                         └───────┬──────────────────────────────────────┘
                                 │
              ┌──────────────────┼───────────────────────────────┐
              ▼                  ▼                                ▼
   ┌────────────────┐  ┌──────────────────┐         ┌───────────────────────┐
 L2│ eval.py        │  │ stats.py         │         │  THE SAFETY/METHOD     │
   │  5 axes +      │  │  rigor_report,   │         │  LAYER                 │
   │  composite()   │  │  wilcoxon, holm, │         │  cast.py               │
   │  (fingerprint) │  │  bootstrap_ci,   │         │  intent_gate.py        │
   ├────────────────┤  │  ordinal_gate    │         │  multi_intent.py       │
   │ geometry.py    │  └──────────────────┘         │  safety_target.py      │
   │  offshell, PR, │                                │  safety_bench.py      │
   │  eff-rank      │   ┌──────────────────┐         │  safety_judge.py      │
   ├────────────────┤   │ judge.py         │         │  gate.py / adversarial│
   │ controls.py    │   │ local_judge.py   │         │  baselines.py         │
   │  shuffled/rand │   │ real_metrics.py  │         └───────────┬───────────┘
   │  stability     │   │ axbench.py       │                     │
   └───────┬────────┘   │ datasets.py      │                     │
           │            └────────┬─────────┘                     │
           └─────────────────────┼───────────────────────────────┘
                                 ▼
   ┌──────────────────────────────────────────────────────────────────────┐
 L1│  PRIMITIVES (touch the model directly)                                 │
   │  hooks.py        SteeringContext, apply_operation (add/relative_add/   │
   │                  rotate/project_out), build_position_mask, probe       │
   │  extract.py      diffmean_vector, pca_top1_vector, collect_activations │
   │  model.py        load_model_cached, get_residual_layers, encode_to_dev │
   │  fakelm.py       FakeResidualLM (offline, deterministic stand-in)      │
   └──────────────────────────────────────────────────────────────────────┘
```

Key rules this graph encodes:

* **L1 primitives never import L2+.** `hooks`/`extract`/`model`/`fakelm` know
  nothing about evaluation, stats, or the CAST method.
* **`eval.py` is the hub of L2** — it owns the five axes *and* the fingerprinted
  `composite()`. Everything that reports a score routes through it.
* **The safety/method layer (CAST)** is a thin policy on top of L1/L2 primitives
  (see DESIGN.md §6): `cast.py` chooses+gates an operation from `hooks.py`, reuses
  `gate.LogisticGate`, `extract.diffmean_vector`, `multi_intent.gram_schmidt`. It
  does **not** re-implement any residual operation.
* **`runner.py` (L3)** is the only writer of the ledger/best_config/reasoning and
  the only trigger of dashboard regeneration. Drivers call *into* it; it never
  imports a driver.
* **`scripts/method_exp_common.py`** is the bridge a *method* driver uses to emit
  one standard ledger row via `runner._log_and_checkpoint` without going through
  the single-vector `run_single_experiment` path.

---

## 2. The numpy ↔ torch seam (the hard invariant)

There is exactly one seam, and it is load-bearing for offline testability and for
running fp32-CPU FakeLM and bf16-CUDA Gemma without branching:

```
   EXTRACTION / CALIBRATION                 INFERENCE (inside a forward pass)
   ────────────────────────                 ────────────────────────────────
   torch forward → .cpu().numpy()           np.ndarray  ──cast──▶  torch.Tensor
   → np.float32 vectors / features          v.to(dtype=h.dtype, device=h.device)
   (extract.py, safety_target.py,           inside hooks.apply_operation /
    gate.condition_features, geometry)       SteeringContext write hook
            │                                         ▲
            ▼                                         │
   numpy math: diffmean, PCA, gram-schmidt,   steering vectors stored as numpy
   logistic gate fit, bootstrap/wilcoxon      on the steerer, converted per call
   (stats.py is pure-numpy/scipy, no torch)
```

* **Everything that touches a model returns `np.ndarray float32`.** Stats,
  controls, geometry, and gate calibration are pure numpy/scipy — they run with
  no GPU and no torch model, which is why `stats.py`/`controls.py`/`eval.py`
  composite math are unit-testable in milliseconds.
* **Everything inside a forward pass is torch.** `hooks.apply_operation` does the
  `v.to(dtype=h.dtype, device=h.device)` cast so a single code path serves FakeLM
  and Gemma. (DESIGN.md §3 documents this seam for the CAST method specifically.)

---

## 3. End-to-end data flow (raw model → composite → dashboard)

```
  raw model (FakeResidualLM | gemma-3-1b-it | gemma-2-2b-it)
     │  model.load_model_cached(name, quant)            [L1]
     ▼
  contrast pairs / AxBench concepts
     │  extract.collect_activations → pos/neg @ layer    [L1, torch→numpy]
     ▼
  steering VECTOR
     │  extract.diffmean_vector / pca_top1_vector        [L2 numpy]
     │  (+ controls.shuffled_label_vector / random_direction for the
     │   matched-displacement DIRECTIONAL control)
     ▼
  STEERED GENERATION
     │  hooks.SteeringContext(model, v, [layer], op, alpha)  [L1 torch]
     │  model.generate(...) inside the context              (greedy gate / sampled eval)
     ▼
  JUDGE  (behavior + fluency)
     │  local_judge.LocalJudge (off-family Qwen)  OR  judge (Gemini API)
     │  fallback: eval.concept_rate lexicon proxy (TAGGED)   [L2]
     ▼
  FIVE AXES                                                  [eval.py, L2]
     │  1 behavior_efficacy   (judge / generation_behavior_scorer)
     │  2 capability_retention (mcq_accuracy / real_metrics)
     │  3 coherence           (perplexity + repetition_rate)
     │  4 safety              (compliance_rate via is_refusal)
     │  5 selectivity         (harmful − harmless refusal gap)
     │  + geometry leading-indicators (geometry.py: offshell, eff-rank, PR)
     ▼
  COMPOSITE  =  eval.composite(metrics)                      [eval.py §6, FINGERPRINTED]
     │  behavior − λ_cap·mmlu_drop − λ_coh·bounded(ΔPPL) − λ_coh_rep·rep
     │           − λ_safe·CR − λ_sel·harmless_refusal − λ_geo·offshell
     │  composite_fingerprint() = sha256(COMPOSITE_FORMULA)[:12]
     ▼
  RIGOR  (stats.rigor_report)                                [stats.py, L2]
     │  paired Wilcoxon + bootstrap CI + Holm + ordinal gate
     ▼
  LOG ROW  → runner._log_and_checkpoint(entry)               [L3]
     │  append experiment_log.jsonl (experiment_num++)
     │  update best_config.json IFF composite beats champion
     │  write reasoning_annotations.json (verdict + learning)
     ▼
  DASHBOARD  dashboard.build_all_dashboards(...)             [L4]
        master index.html + per-hypothesis + per-experiment pages
```

### 3a. Composite-field integrity (improvement #45)

The `composite` column in `experiment_log.jsonl` may hold **only** the output of
`eval.composite()` (the fingerprinted 5-axis formula) — never a raw behavior
mean. Two writer paths enforce this:

* **`runner.run_single_experiment`** always computes the full 5 axes via
  `eval.evaluate_bundle`, so its rows always carry a real composite.
* **`scripts/method_exp_common.log_method_experiment`** is for *method* drivers
  whose primary result is a single raw metric (a delta, a best behavior, a
  cosine, a Gram mass, a retention ratio). These do **not** compute all five
  axes, so they pass `composite=None`; the raw value lives in `method_value`. The
  row then records `composite_is_real=False` and stores JSON `null` in
  `composite`. `runner._log_and_checkpoint` treats `None` as "no composite" — it
  can never win the champion slot and never crashes the verdict renderer. The
  dashboard's `_num()` helper renders a `None` composite as its default (not a
  number). Pass a real composite **only** when all five axes were measured and
  fed through `eval.composite()` (e.g. `scripts/run_safety_eval.py`).

---

## 4. Where state lives (the ledger, CLAUDE.md §12)

All durable research state is JSON under `autoresearch_results/`, written
exclusively by `runner.py` / `method_exp_common.py`:

| File | Writer | Role |
|---|---|---|
| `autoresearch_results/experiment_log.jsonl` | `_log_and_checkpoint` | **append-only** experiment history; one row per experiment; `experiment_num` auto-increments |
| `autoresearch_results/best_config.json` | `_log_and_checkpoint` | global champion (updated iff composite strictly beats prior best) |
| `autoresearch_results/reasoning_annotations.json` | `_write_reasoning` / `_author_reasoning` | per-experiment 7-step entries (pre-run authored before launch; verdict+learning post-run) |
| `autoresearch_results/running.json` | `runner` | transient "an experiment is running" signal |
| `ideas/_campaigns/*.json` | `write_campaign` | full per-campaign artifacts (curves, grids, rigor reports) |
| `ideas/<NN>/…` | drivers / dashboard | per-hypothesis sub-project + sub-dashboard |
| `dashboard/` + `docs/dashboard/` | `dashboard.build_all_dashboards` | generated HTML (master + per-hyp + per-exp); never hand-edited |
| `IDEA_TABLE.md`, `EXPERIMENT_LEDGER.md`, `FINDINGS.md` | human + agents | hypothesis registry, promotion log, rigor-gated findings |

**Invariants:** the JSONL is append-only (never rewritten — historical rows with
the old mislabeled composite stay as-is; only *future* rows are honest). The
composite formula string is SHA-256 fingerprinted; editing it changes
`composite_fingerprint()` and breaks cross-run comparability by design.

---

## 5. How to add a new experiment driver

Two shapes, depending on whether your result is a steering-vector evaluation or a
method-level metric.

**A. Single-vector evaluation (full 5 axes).** Use the runner directly:

1. Build/extract your vector with `extract.diffmean_vector` (numpy).
2. Call `runner.run_single_experiment(config, model, tokenizer, ...)` — it runs
   the 5 axes through `eval.evaluate_bundle`, computes the real composite, and
   appends the row. Author the pre-run reasoning entry first (no `--bypass`).

**B. Method experiment (primary metric is not a single vector's 5 axes).** Use
`scripts/method_exp_common.py` (the pattern in `run_axbench_e*`, `confirm_e7*`,
`run_e15/e20/e45`):

1. `import method_exp_common; method_exp_common.LOGGING_ENABLED = not args.no_log`
   (so `--quick`/`--no-log` offline smoke does not pollute the append-only ledger).
2. Compute your metric and a `rigor_report` if you make any comparative claim.
3. `write_campaign(tag, payload)` for the full artifact.
4. Author a genuine `reasoning` dict (diagnosis / citations / hypothesis /
   prediction) — `_author_reasoning` writes it before the row is appended.
5. Call `log_method_experiment(..., method_value=<raw metric>, composite=None,
   behavior_efficacy=<honest behavior>, ...)`.
   **Pass `composite=None` unless you actually computed all five axes through
   `eval.composite()`** (improvement #45). Put the raw primary metric in
   `method_value` / `method_extra`, and add the one-line docstring note:
   *"composite field now holds only the fingerprinted 5-axis composite; raw
   behavior metric is in method_value."*

In both cases: ONE config change per experiment, start from the champion, climb
the ladder rung-by-rung (CLAUDE.md §4–5), commit before the next launch.

---

## 6. Offline-testability story (FakeResidualLM)

Gemma is gated on HuggingFace and may be unavailable (no token, no license, no
network, or no GPU). The repo is therefore **fully exercisable offline** via
`fakelm.FakeResidualLM` (`src/steering/fakelm.py`):

* `model.load_model_cached("fake")` returns a `FakeResidualLM` + tokenizer with
  the **same interface** real Gemma exposes: `get_residual_layers`, a forward
  returning `.logits`, and the residual hook points `hooks.py` attaches to. So
  every L1 primitive (extract, hooks, probe) and every L2 metric runs unchanged.
* **No `.generate`.** FakeLM emits gibberish logits, so `eval._can_generate`
  returns False for it and the generation-based scorers fall back to a
  deterministic, **tagged** path: `generation_behavior_scorer` → projection
  proxy; `generate_responses` → `fake_safety_responses` (a fixed refusal, the
  CR≈0 no-leak baseline). The `safety_real=False` / `scorer="projection"` tags
  record that a placeholder ran, so a FakeLM result can never be mistaken for a
  real measurement.
* **Determinism.** FakeLM is seeded and reproducible, so UNIT/SMOKE rungs
  (CLAUDE.md §4) gate plumbing (vector changes logits; state restores exactly;
  conditional gate is a NO-OP when no intent fires — DESIGN.md §5) in seconds,
  with no model download. Drivers expose `--quick`/`--no-log` to run this path
  without writing the append-only ledger.
* The pure-numpy layers (`stats.py`, `controls.py`, `geometry.py`, the
  `eval.composite` math) need no model at all and are tested directly.

This is what makes "dozens of iterations/day on a 16 GB laptop" possible: the
cheap, model-free and FakeLM paths catch bugs long before any real-Gemma rung
spends GPU.

---

## 7. File-reference quick index

| Concern | File |
|---|---|
| Residual operations + hooks | `src/steering/hooks.py` |
| Vector extraction (DiffMean/PCA) | `src/steering/extract.py` |
| Model loading + layer access | `src/steering/model.py` |
| Offline stand-in model | `src/steering/fakelm.py` |
| Five axes + composite (fingerprint) | `src/steering/eval.py` |
| Geometry leading-indicators | `src/steering/geometry.py` |
| Statistical rigor contract | `src/steering/stats.py` |
| Matched-displacement controls | `src/steering/controls.py` |
| Off-family judges | `src/steering/judge.py`, `local_judge.py` |
| Real benchmarks (MMLU/etc, AxBench) | `src/steering/real_metrics.py`, `axbench.py`, `datasets.py` |
| CAST method (conditional, multi-intent) | `src/steering/cast.py`, `intent_gate.py`, `multi_intent.py`, `safety_target.py` (+ DESIGN.md) |
| Safety bench / judge / adversarial | `src/steering/safety_bench.py`, `safety_judge.py`, `adversarial.py`, `gate.py`, `baselines.py` |
| Experiment loop + ledger writer | `src/steering/runner.py` |
| Method-experiment ledger bridge | `scripts/method_exp_common.py` |
| Dashboard generation | `src/steering/dashboard.py` |
| Drivers | `scripts/run_axbench_*.py`, `scripts/confirm_e7*.py`, `scripts/run_e15/e20/e45.py`, `scripts/run_safety_eval.py` |
| Durable state | `autoresearch_results/*.json(l)` |

---

*Companion docs: `CLAUDE.md` (constitution: ladder, composite, rigor floor),
`src/steering/DESIGN.md` (CAST method spec), `AUTORESEARCH_PROCESS.md` (the loop
in detail).*
