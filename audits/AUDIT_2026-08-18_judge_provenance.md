# AUDIT — judge provenance: which lessons actually need a GPU re-run

**Date:** 2026-08-18
**Scope:** every lesson under `steering_tutorials/`
**Method:** read `hello_world_steering/judge.py` (the shared `Judge`), then for each
lesson read the runner to see whether an LLM's *text* is graded, and read the
on-disk `artifacts/*.json` to see what provenance was actually recorded. No model
was loaded and no job was run (the 4090 is busy).
**Rule under test:** CLAUDE.md §17 rubric item 3 — an off-family judge
(`STEER_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct`) is mandatory for every reported
GENERATION number.

**Internal QA pass — independent external review pending** (same-model-family
circularity disclosure, CLAUDE.md §14).

---

## 0. Two findings that reframe the whole question

### 0.1 `GIBBERISH` is judge-FREE in every lesson that uses `Judge`

`judge.py:271-297` runs the deterministic `is_gibberish()` gate (`judge.py:68-102`)
**before** any model call. It is pure string statistics — distinct-token ratio and
run-length — with no model involved. Therefore in *every* lesson built on this
`Judge`, `gibberish_rate` is a judge-free measurement. Only the
REFUSAL-vs-COMPLIANCE split consults a model.

Consequence: coherence-collapse conclusions do not need a re-run. The
`realignment` falsification (189 GIBBERISH of 191) and the `hello_world_steering`
finding that the vector "destroys coherence" both rest on the deterministic gate,
not on the judge.

### 0.2 Self-judging is a *conservative* bias for null-refusal claims

The documented failure mode is that a 1B model grading its own output **inflates**
refusal (`judge.py:134-137`). So for any claim of the form "refusal did **not**
rise", an unknown-provenance number that may be self-judged is biased *toward the
opposite* of the claim. Those claims survive the uncertainty; claims of the form
"refusal **did** rise" or "ASR **fell**" do not. This asymmetry is what separates
priority 2 from priority 5 in §3.

---

## 1. No artifact on disk carries the full stamp

`Judge.stamp()` (`judge.py:250-269`) emits four keys:
`judge_id`, `is_self_judge`, `judge_model_id`, `off_family`. A scan of all 47
`*/artifacts/*.json` files finds **zero** artifacts carrying all four. The
stamping code is wired into 16 runners (`**judge.stamp()`) but **no artifact has
been regenerated since**. Every provenance statement below therefore rests on the
older, weaker legacy key — and on whether that key was read off the `Judge`
object or copied from config.

| what is on disk | lessons |
|---|---|
| legacy `judge` / `judge_id` / `judge_model` string, read from the `Judge` object via `getattr(judge, "judge_id", ...)` | `curveball`, `talan`, `contextual_steering`, `fine_grained`, `flas` |
| hand-built nested `judge` dict incl. `off_family: true` | `realignment` |
| explicit `"judge": null` + a written N/A note | `meerkat`, `multiturn_jailbreak`, `trajguard`, `cross_trajectory`, `biencoder_guard` |
| explicit nulls in every stamp slot (an honest UNKNOWN) | `reft_r1/compare_bases.json` |
| `judge_id: "self"` | `gavel` |
| **no judge key at all** | the eight UNKNOWNs |

The legacy string is genuine but second-tier evidence: `getattr(judge, "judge_id")`
can only read `Qwen/Qwen2.5-3B-Instruct` if `STEER_JUDGE_MODEL` was set in that
process (`judge.py:161-169, 226-243`). It is not fabricable from a README. It is
still not the stamp, and per §18.8 ("stamp your inputs") these artifacts predate
the code beside them.

---

## 2. Per-lesson classification

Class **A** = detection (classifier vs ground-truth labels, nothing generates and
nothing grades text — rule 3 genuinely N/A).
Class **B** = generation, graded (rule 3 applies).
Class **C** = mixed.

### Class A — DETECTION. NO RE-RUN, EVER.

| lesson | evidence | verdict |
|---|---|---|
| `hello_world` | `train_probe.py`, `cross_validate.py`, `eval_ood.py` contain **zero** occurrences of "judge"; `artifacts/metrics.json` reports `roc_auc` / `confusion_matrix` against JailbreakBench labels | **NO RE-RUN — detection** |
| `probe_tuning` | `sweep_layers.py`, `sweep_mlp.py` — no judge; `artifacts/sweep_mlp.json` is CV ROC-AUC | **NO RE-RUN — detection** |
| `meerkat` | `run_meerkat.py:576-580` writes `"judge": None` with an explicit N/A comment; metrics are AP / clustering purity | **NO RE-RUN — detection** |
| `cross_trajectory` | `run_cross_trajectory.py` — no judge symbol at all; `results.json` has `"judge": null`, metrics are fold AUC + OOD | **NO RE-RUN — detection** |
| `biencoder_guard` | `run_biencoder_guard.py` — no judge symbol; `results_CAUSAL_ENCODER_SUSPENDED.json` `"judge": null`; metrics are seen/held-out AUC + `confound` | **NO RE-RUN — detection** |
| `multiturn_jailbreak` | `run_multiturn.py:651-653` writes `"judge": null` **plus** a `judge_note` spelling out why rule 3 is N/A — the best-documented instance in the repo | **NO RE-RUN — detection** |
| `trajguard` | `run_trajguard.py:624` `"judge": None`. **It does generate text** (`token_trajectories.json` holds `prompts` + `completions`) — but nothing *grades* it; a classifier reads the token trajectory and is scored against ground-truth source labels. Generation alone does not trigger rule 3 | **NO RE-RUN — detection** |

### Class A* — DETECTION, but with a *label*-provenance question rule 3 does not cover

| lesson | evidence | verdict |
|---|---|---|
| `streaming_trajectory_aggregation` | **Absent from the prior inventory entirely.** `run_sta.py` / `embed.py` contain no judge; the metric is fold AUC. But `types.py:143` records `label_provenance: "gemini-3.1-pro-preview judge log"` and `evaluate.py:307` / `types.py:214` carry `oracle_note: "lead measured against a judge oracle, NOT ground truth"`. So the *labels*, not the outputs, come from a judge — an off-family one (Gemini vs Gemma), so rule 3 is satisfied as written | **NO RE-RUN — detection.** But `lead_time` is oracle-relative, not ground-truth-relative, and should not be quoted as if it were |

### Class B — GENERATION, fully judge-dependent headline

| lesson | evidence | provenance | verdict |
|---|---|---|---|
| `gavel` | `run_gavel.py:278-298` grades pass-through generations; `artifacts/results.json` → `passthrough.judge_id = "self"`, `harmful_passed_compliance_rate 0.2428`, `benign_passed_answered_rate 0.4481` | **CONFIRMED SELF-JUDGED** | **RE-RUN REQUIRED** (partial — see §2 note) |
| `rogue_scalpel` | `run_rogue_scalpel.py:52-58` defines `asr = verdicts.count("COMPLIANCE")/n`; the entire `ladder` (baseline `asr 0.21`, `refusal_rate 0.70`) is judge-scored. `results.json` has no judge key | UNKNOWN | **RE-RUN REQUIRED** |
| `multi_intent` | `run_multi_intent.py:320-330` judges every concept's eval prompts; `ladder[k].raw.success` / `.crosstalk` are judge-derived. `results.json` keys carry no `judge_id` despite the runner now writing `**judge.stamp()` at :404 | UNKNOWN | **RE-RUN REQUIRED** |
| `non_identifiability` | `run_nonident.py:262-296` judges each candidate direction; `nonident.verdict` ("SUPPORTED (screening): 4 directions … reach within 80% of best refusal") rests entirely on `refusal_rate`. `results.json` has no judge key | UNKNOWN | **RE-RUN REQUIRED** |
| `hello_world_steering` (main arm) | `run_steering.py:586, 664-665` judges every alpha cell; `unconditional[*].refusal_rate` and `conditional.{harmful_refusal_rate, benign_over_refusal_rate}` are judge-scored. `results.json` has **no** `judge` key even though `run_steering.py:425` writes one → the artifact predates that line | UNKNOWN | **RE-RUN REQUIRED** (but see §0.2 — the null is conservative) |

### Class C — MIXED. PARTIAL: only the named metrics need a re-run.

| lesson | judge-FREE (no re-run) | judge-DEPENDENT (re-run) | verdict |
|---|---|---|---|
| `prompt_activation_duality` | `duality.*` — `cos_refusal_shift_vs_vector`, `cos_control_shift_vs_vector`, `random_cosine_baseline`, `*_shift_norm`. Pure geometry; this **is the lesson's headline** | `effect[*].refusal_rate` / `compliance_rate` for unsteered / residual / attention (`run_duality.py:240, 265-267`) | **PARTIAL — only `effect[*].refusal_rate` and `compliance_rate`** |
| `decomposing_prompting` | `decomposition.*` — `mean_abs_proj`, `mean_residual_norm`, `on_direction_frac`, `consistency`, `shared_translation_frac`. Geometry only | `recovery.{baseline,prompting,steer_proj,steer_shared}_refusal` and both `recovery_*` figures (`run_decompose.py:252-255`) | **PARTIAL — only the `recovery` block.** Note `recovery_proj = recovery_shared = 0.0` is a *null*, and steered refusal (0.26) sits *below* baseline (0.33), so the null is robust to a refusal-inflating judge |
| `reft_r1` | `detection.{reft_r1_auc 0.5678, diffmean_auc 0.7477}` — probe AUC vs ground-truth labels. This carries the AxBench "simple baselines are strong" claim | `steering.*` refusal/gibberish rates, and all of `compare_bases.json` (whose stamp is explicitly all-`null`) | **PARTIAL — only the `steering` arm.** `run_reft.py:562-567` now **hard-fails** if `resolve_judge_id() == SELF_JUDGE_ID`, so any future run is safe by construction; the legacy abliterated artifact is already quarantined as `results_abliterated_LEGACY_UNKNOWN_JUDGE.json` |
| `gavel` | block rates, per-CE firing rates, `broad_baseline` (rule-based, no model grading) | `passthrough.harmful_passed_compliance_rate`, `passthrough.benign_passed_answered_rate` | **PARTIAL — only the `passthrough` block.** `gavel/rejudge.py` exists and is CPU-verified |
| `stacking` | — | `results.json` `rungs[*]` refusal/gibberish rates and `decision.*` | **PARTIAL / LOW.** `results.json` is UNKNOWN, but `hillclimb_results.json` `meta.judge_id = "Qwen/Qwen2.5-3B-Instruct"` is off-family, and `results.json`'s own `decision.verdict` is already **"INCONCLUSIVE at this scale"** — a re-run cannot move a verdict that claims nothing |

### Off-family, no re-run

| lesson | evidence |
|---|---|
| `realignment` | `results.json` → nested `judge: {judge_id: "Qwen/Qwen2.5-3B-Instruct", off_family: true}` (`run_realignment.py:811-814`). The strongest provenance record in the repo. Also `run_realignment.py:603-627` actively guards the second-instrument phase against a stale env var |
| `fine_grained` | `results.json` top-level `judge_id: "Qwen/Qwen2.5-3B-Instruct"`, written from `**judge.stamp()` (`run_fine_grained.py:276`) |
| `curveball` | `results.json` `judge: "Qwen/Qwen2.5-3B-Instruct"` from `getattr(judge, "judge_id", "self")` (`run_curveball.py:273`) |
| `talan` | `results.json` `judge: "Qwen/Qwen2.5-3B-Instruct"` (`run_talan.py:361`) |
| `contextual_steering` | `results.json` `judge: "Qwen/Qwen2.5-3B-Instruct"` (`run_contextual.py:457`) |
| `flas` | `judge_model: "Qwen/Qwen2.5-3B-Instruct"` (`run_flas.py:607`) — **but the only artifact on disk is `results_v2_SUPERSEDED_DIVERGED.json`, self-flagged `SUPERSEDED_BY_V3`, and no v3 artifact exists.** Provenance is fine; the *headline* has no backing artifact. This is a missing-run defect, not a judge defect |

---

## 3. Prioritised re-run list

Ordered by how far the conclusion moves if the judge turns out to have been the
1B grading itself. Everything above the line has a judge-scored rate as its
**headline**.

1. **`gavel`** — the only *confirmed* violation (`judge_id: "self"`). Headline
   pass-through rates (compliance 0.243, benign-answered 0.448) are exactly the
   numbers a self-judge distorts. `rejudge.py` is written and CPU-verified; this
   is the cheapest large correction available.
2. **`rogue_scalpel`** — the lesson *is* the ASR ladder. Every rung is
   `count("COMPLIANCE")/n`. There is no judge-free arm to fall back on, and the
   §10 Rogue Scalpel mandate makes this the most consequential number in the
   course. Fold in the unrun `negative_add` mode and the A–E guard ablation
   (SESSION STATE §18.6 item 3) so one GPU window buys both.
3. **`multi_intent`** — `success` and `crosstalk` across the whole ladder are
   judge-derived, with no judge-free arm. The `cosine_matrix` survives; nothing
   else does.
4. **`non_identifiability`** — the `SUPPORTED` verdict turns on a **refusal
   spread of 0.0375** across four directions. That is well inside plausible judge
   noise for a 0.75-AUC instrument, so this verdict is the most fragile in the
   set relative to its confidence. The alpha sweep is already wired and unrun
   (§18.6 item 3) — run both together.
5. **`hello_world_steering`** (main arm) — judge-dependent, *but* §0.2 applies:
   the headline is "refusal rises NOWHERE", and a self-judge inflates refusal, so
   the claim is stated against its own bias. The number that genuinely needs the
   off-family judge is `conditional.benign_over_refusal_rate = 0.48`, which is
   *upward*-biased by self-judging and is quoted as a cost of the method.
6. **`reft_r1`** (steering arm only) — the AxBench headline lives in the
   judge-free `detection` AUCs and stands. Re-run buys the steering rates and a
   non-null `compare_bases.json` stamp.
7. **`decomposing_prompting`** (`recovery` only) — a null whose bias direction
   works against the null; the geometry headline is judge-free.
8. **`prompt_activation_duality`** (`effect` only) — the duality cosines are the
   headline and are judge-free.
9. **`stacking`** (`results.json` only) — verdict is already INCONCLUSIVE and the
   hill-climb sibling is off-family. Lowest value in the set.

**Zero-GPU follow-up, independent of the above:** no artifact anywhere carries
the four-key `stamp()` (§1). Any re-run should be treated as the moment the stamp
finally lands on disk, and a build-time gate should refuse to publish a
generation number from an artifact missing `off_family` — per §18.8, "a check
that must be remembered will eventually be skipped."

---

## 4. Disagreements with the prior inventory

1. **`streaming_trajectory_aggregation` is missing from the inventory entirely.**
   It is detection (no re-run), but it has a *label*-provenance property rule 3
   does not describe: labels come from a Gemini judge log
   (`types.py:143`) and `lead_time` is oracle-relative (`evaluate.py:307`).
2. **Four of the eight "UNKNOWN" lessons are MIXED, not blocked.**
   `prompt_activation_duality`, `decomposing_prompting`, and `reft_r1` each have a
   judge-free arm carrying the headline; `stacking` already has an off-family
   sibling artifact and an INCONCLUSIVE verdict. Treating all eight as equally
   blocked overstates the debt by roughly half.
3. **`gavel` is MIXED, not wholly poisoned.** Block rates and per-CE firing are
   judge-free; only the `passthrough` block is self-judged.
4. **`reft_r1` is no longer re-runnable-into-the-same-hole.**
   `run_reft.py:562` hard-fails on a self-judge, so its future provenance is
   guaranteed by code rather than by discipline. Worth porting to the other seven.
5. **The "OFF-FAMILY VERIFIED" tier is weaker than it reads.** None of the six
   carries the full stamp — five carry a legacy string and one a hand-built dict
   (§1). The string is real evidence (it can only come from a `Judge` whose env
   was set) but it is not the stamp, and those artifacts predate the code beside
   them.
6. **`flas` has a provenance-clean artifact that is self-declared SUPERSEDED,
   with no successor on disk.** Its v3 headline is unbacked. That is a separate
   and arguably worse defect than an unknown judge.
7. **`trajguard` generates text but is still class A** — nothing grades the text;
   a classifier reads the trajectory. Generation alone does not trigger rule 3.

---

## 5. Addendum (lead, 2026-08-18) — STA label provenance, resolved precisely

The audit flagged that STA's labels come from a Gemini judge log
(`types.py:143`) and that this is a provenance axis rule 3 does not cover. Correct
that it is uncovered, but the blast radius needed pinning down, because
`Corpus.label_provenance` reads as if it describes the *trajectory* labels:

    label_provenance: "gemini-3.1-pro-preview judge log (LLM-generated, not human)"

Traced it. There are **two different label populations** and they have **different
provenance**:

| population | source | judge-free? |
|---|---|---|
| **trajectory** label (pos/neg) | the dataset's own `ground_truth` 0/1 field, `corpora/__init__.py:485` | **YES** |
| **per-step** labels (`ideal_flagging_step`, `point_of_no_return_step`) | Gemini-3.1-Pro judge log | **NO** |

Consequences, and they cut opposite ways:

- **F1 and F3 are unaffected.** The whole main AUC ladder — mean_pool 0.7397 vs
  max_pool 0.8581 vs gru 0.9051, and best-vs-last_step — is scored against
  trajectory `ground_truth`. The mean-pooling collapse stands as a judge-free
  result. Reading `label_provenance` alone would wrongly discredit it.
- **F5 gets worse, not better.** Lead time is measured against the Gemini
  *oracle* step, so it is not "steps of warning before harm" but "steps before a
  judge would have flagged it". `shade.py:44-52` states this outright: *"lead time
  is measured against a judge ORACLE, never against truth."* Stacked on the
  already-recorded hollowness (3 detections of 132 eligible, 0.294 false alarm),
  **F5's HOLD should not be quoted at all** without both caveats attached.

**Defect to fix:** the field name `label_provenance` is singular and sits on
`Corpus`, so it reads as covering the labels the ladder is scored against — which
it does not. It should be split (`trajectory_label_provenance` vs
`step_label_provenance`) or renamed, because as it stands the honest string
invites the wrong inference in both directions.
