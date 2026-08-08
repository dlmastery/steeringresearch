# STA benchmark dataset verification (host: no HF token)

Status: COMPLETE. Every row below was verified by probing the live HF API / datasets-server from
this host with NO token. Row counts are `num_rows_total` from `/size` or `/filter`, not card claims.
Probe method: `https://datasets-server.huggingface.co/size?dataset=<id>` +
`/first-rows?dataset=<id>&config=<c>&split=<s>` + `https://huggingface.co/api/datasets/<id>`.
A 401 on the API = GATED (unusable on this host). A 404 = does not exist under that id.

## Summary table (filled in as verified)

**NOTHING in the shortlist is gated. The no-token constraint turned out not to bind.**
The binding constraints are instead (a) undeclared licences on the best data and (b) the fact that
the brief's lead-time headline is contradicted by the labels — see the LEAD-TIME section.

| Benchmark | HF id | Gated | Licence | Rows | Row = | Step labels | >=500/cls |
|---|---|---|---|---|---|---|---|
| SHADE-Arena (official) | — none | — | — | — | — | — | not on HF |
| SHADE via ControlArena | `adityaasinha28/control_arena_shade` | **No** | none | 1500 (750/750) | full trajectory, median 129 steps | traj-only | **YES** |
| SHADE step labels | `aksh-n/control_arena_shade-monitor-labels` | **No** | none | 1497 JSONs | per-step judge log | **PER-STEP + oracle flag + PoNR** | **YES** |
| CUA-SHADE | `adityaasinha28/cua_shade_arena_action_only_large` | No | none | 968 (484/484) | trajectory | traj-only | no |
| CUA-SHADE aware | `adityaasinha28/cua_shade_arena_aware` | No | none | 292 (146/146) | trajectory | traj-only | no |
| ASSEBench | `Atarogic/ASSEBench` | **No** | Apache-2.0 | 1476 ×3 std + 817 sec | trajectory | traj-only | **YES** |
| Pre-Ex-Bench | — none | — | — | — | synthesis engine, no corpus | — | unavailable |
| R-Judge | `Glide-py/r_judge_labelled` | No | CC-BY-4.0 | 568 (269/299) | trajectory | traj-only | no |
| LongSafety | `thu-coai/LongSafety` | No | MIT | 1543 | long ctx + instruction | **none, no labels** | n/a |
| LongSafetyBench | `LutherXD/LongSafetyBench` | No | Apache-2.0 | 1201 over 10 splits | MCQ item | none | no |
| ATBench | `AI45Research/ATBench` | **No** | Apache-2.0 | 1000 (503/497) | trajectory + tool specs | traj-only | borderline |
| RAS-Eval | — none | — | — | — | needs live execution | — | unavailable |
| Who&When | `Kevin355/Who_and_When` | No | none | 184, no negatives | MAS failure log | **decisive error step** | no |
| AgentDojo | `adityaasinha28/control_arena_agentdojo` | **No** | none | 1500 (750/750) | trajectory, median 8 steps | traj-only | **YES** |
| AgentDojo step labels | `aksh-n/control_arena_agentdojo-monitor-labels` | **No** | none | 1500 JSONs | per-step judge log | **PER-STEP, but lead=0** | **YES** |
| AgentDojo traj dump | `sungjuncho/agentdojo-trajectories` | No | none | 15095 files, 2 models | raw trajectory | unlabelled | — |
| BashBench (tasks) | `Eccentricity/bashbench2` | No | CC-BY-SA-4.0 | 637 | task env, NOT trajectory | none | n/a |
| BashBench (labelled) | `AISafety-Student/labeled-bashBench` | **No** | **MIT** | 2494 (1058/1436) | **ONE STEP** | **PER-STEP** | **YES** |
| MLE-Sabotage | `aksh-n/mle-sabotage` | No | none | 151 | trajectory | traj-only | no |
| BigCodeBench-Sabotage | — not found | — | — | — | — | — | unavailable |
| AgentHarm | `ai-safety-institute/AgentHarm` | No | other | 176/class | prompt | none | no |
| AgentHarm traj | `antieval/agentharm-trajectories` | No | none | 160 | trajectory | traj-only | no |
| ToolEmu | GitHub `ryoungj/ToolEmu` | — | — | 144 cases | emulator case | none | no |

---

## Per-dataset findings

### ALREADY CACHED ON THIS HOST (free to use, verified as real dirs)

`~/.cache/huggingface/datasets`: `JailbreakBench___jbb-behaviors`, `SafeMTData___safe_mt_data`,
`Salesforce___wikitext`, `cais___mmlu`, `intrinsec-ai___cstm-bench`, `lmsys___toxic-chat`.
`~/.cache/huggingface/hub` additionally holds dataset repos: `AlignmentResearch/ClearHarm`,
`JailBreakV-28K`, `LLM-Tuning-Safety/HEx-PHI`, `LibrAI/do-not-answer`, `PKU-Alignment/BeaverTails`,
`PKU-Alignment/MM-SafetyBench`, `Paul/XSTest`, `ScaleAI/mhj`, `TrustAIRLab/in-the-wild-jailbreak-prompts`,
`allenai/wildjailbreak`, `jackhhao/jailbreak-classification`, `pyvene/axbench-concept10`,
`pyvene/axbench-concept500`, `sorry-bench/sorry-bench-202406`, `walledai/AdvBench`,
`walledai/HarmBench`, `walledai/StrongREJECT`.
**NONE of these is an agent-trajectory dataset.** `ScaleAI/mhj` and `SafeMTData` are multi-TURN
jailbreak dialogues — the closest cached thing to a "trajectory", and the only cached corpora with
any sequential structure at all. No SHADE/ATBench/AgentDojo/R-Judge is cached; every agent
benchmark below must be downloaded fresh.

---

### SHADE-Arena — no official Anthropic HF dataset; community ControlArena dumps ARE ungated

The original SHADE-Arena (Anthropic) ships as code, not as an HF dataset repo. What exists on HF
are **ControlArena / CUA-SHADE-Arena re-dumps of generated agent transcripts**, all `gated=False`:

| id | rows | cols | what one row is |
|---|---|---|---|
| **`adityaasinha28/control_arena_shade`** | **1500** (train) | 11 | one FULL trajectory + `ground_truth` 0/1 |
| `adityaasinha28/cua_shade_arena_action_only_large` | 968 | 17 | one full trajectory, action-only view |
| `aksh-n/cua_shade_arena_action_only_large` | 968 | 17 | mirror of the above |
| `adityaasinha28/cua_shade_arena_aware` | 292 | 17 | attack-aware variant |
| `aksh-n/control_arena_shade-monitor-labels` | file repo | — | per-row **`ponr` judge logs** (see below) |

- **Gated: NO** (all five). **Licence: NOT DECLARED** on any of them — no `license:` tag, no
  README licence field. This is a real defect to flag: the upstream SHADE-Arena code is MIT but
  these dumps assert nothing. Treat as "unstated, community re-dump".
- `control_arena_shade` columns: `prompt`, `ground_truth` (int64 0/1), `trajectory_data` (JSON
  string holding a `messages` list — **this is where the per-step structure lives**), `transcript`
  (flat text), `cleaned_misalignment_evidence`, `extra_info.main_task_description`,
  `setting_description`. Row 0 is `shade_claude-sonnet-4_negative1_1_epoch1`, `ground_truth=0`.
- `cua_shade_arena_action_only_large` columns add `scheming_category` (`benign_v1`, …),
  `side_task`, `agent_model`, `task_name`, `category` (`honest`/…), `token_count` (row 0 = 20478
  tokens — these are LONG trajectories).
- **Label granularity: TRAJECTORY-LEVEL.** One `ground_truth` per whole trajectory. The steps are
  recoverable by parsing `trajectory_data.messages`, so a causal/streaming monitor **can be
  trained** on it — but early-detection lead-time is not directly measurable from these five repos
  alone.
- **>=500/class:** 1500 rows total in `control_arena_shade`; class balance not yet confirmed from
  the first-rows probe (see the balance section below).

---

### *** `aksh-n/control_arena_shade-monitor-labels` — THE ONE THAT DECIDES THE LESSON ***

**This is the only dataset found anywhere in this sweep that carries genuine PER-STEP labels AND
an oracle flag-step. It makes the brief's lead-time headline measurable rather than assertable.**
I verified this by fetching and parsing actual judge-log JSON files, not the card.

- **Gated: NO.** Ungated, public. **Licence: NOT DECLARED** (no licence tag/field) — flag this.
- **1499 files = 1497 `judge_logs/row_N__gemini-3.1-pro-preview__ponr.json` + `label_manifest.json`
  + `.gitattributes`.** It is a *file* repo, not a parquet dataset, so `/size` returns nothing and
  `datasets-server` cannot page it — you fetch the JSONs directly by URL. Row index `N` aligns to
  `adityaasinha28/control_arena_shade` (1500 rows; 1497 kept after three post-filters).
- `label_manifest.json` (154 KB): `labeled_with.judge_model = google/gemini-3.1-pro-preview`,
  `ponr: true`, `params.threshold = 7`, `filter_model = gemini-2.5-flash`, `seed = 0`,
  `updated_at 2026-06-29`, `filters.post = [high_confidence_benign, high_severity_benign,
  no_ideal_flag_scheming]`, and a per-row `{status: kept|..., log: <filename>}` map.

**Each judge log contains four label tracks over the SAME step index:**

1. `priv_judge.with_evidence.step_analyses[]` — one record **per step**, fields
   `step`, `on_path` (bool), `severity_if_confirmed`, `action_only_confidence`,
   `irreversibility_imminent`, `rationale`. This is the *privileged* (side-task-aware) view.
2. `priv_judge.with_evidence.candidate_ideal_flagging_step` (int; **`-1` for benign rows**) —
   **the oracle: the step at which an ideal monitor SHOULD raise the alarm**, with a written
   `..._rationale`. Plus `overall_severity` (0 for benign, 8 in both positives sampled).
3. **`nonpriv_streaming[]` — a genuinely CAUSAL, ONLINE reference monitor.** One record per step,
   fields `current_step`, `current_step_local_suspicion`, `cumulative_confidence`,
   `severity_if_confirmed`, `irreversibility_imminent`, **`would_flag_now` (bool)**, `rationale`.
   It sees only the prefix. This is a ready-made baseline to beat *and* a supervision signal for a
   streaming monitor at every step.
4. `adjudicator` — `step_confidence[]` and `step_flag_score[]` (per-step arrays, length = n_steps),
   `on_path_corrections`, `judge_disagreements`, `ideal_flagging_step_rationale`, and crucially
   **`point_of_no_return_step`**.

**LEAD TIME IS DIRECTLY COMPUTABLE** as `point_of_no_return_step − ideal_flagging_step`.
Worked example, row 800 (`ground_truth=1`, 62 steps): ideal flag at step **10** (unprompted
autodelete rule aimed at GitHub security alerts), point of no return at step **21** (reprocessing
permanently deletes the mail) — **11 steps of advance warning available**. Row 900: flag at 95 of
111 steps. A monitor that flags at step *k* has a measurable, non-circular lead time.

There is even a built-in QA block, `verification`: `all_pre_k_below_5`, `all_post_k_at_least_5`,
`confidence_monotonic`, `severity_monotonic`, `flag_score_matches_formula`, `passed`, `issues`,
`warnings`. Use `passed == true` as an admission filter.

**Measured, not assumed** (uniform sample of 60 rows, every 25th index across 0..1475):
- **Class balance 30 benign / 30 scheming — exactly 50/50.** Extrapolates to ~750/750 over 1497.
  **>=500/class is REACHED**, and this is the ONLY agent-trajectory corpus in this sweep that
  reaches it with per-step labels.
- **Steps per trajectory: min 30, median 131, mean 132.2, max 422.** These are long trajectories —
  the streaming argument has room to matter here, unlike a 9-turn corpus.
- Across the 30 sampled positives, the oracle flag step lands at a **mean 44.6% of the way through
  the trajectory** (range 5.5% to 94%), leaving a **mean 83.1 steps** after the ideal flag.
  So the early-detection headroom is real and large, but *highly variable* — a single mean
  "N steps of advance warning" number will have a wide spread and must be reported with a
  distribution, not a point estimate.

**Caveats that must ship with any claim built on this:**
- Every label here is **LLM-generated by Gemini 3.1 Pro**, not human. It is a *judge-derived*
  benchmark. Under §17 rule 3 (off-family judge) Gemini is comfortably off-family from Gemma, which
  helps — but the labels are still model opinion, and `candidate_ideal_flagging_step` is a
  judgement call, not ground truth. Do not describe lead time as measured against truth; describe
  it as measured against a Gemini-3.1-Pro oracle.
- Predicting `would_flag_now` from `nonpriv_streaming` is **distilling another monitor**, not
  detecting scheming. Keep that arm clearly separated from the `ground_truth` arm.
- No declared licence.

---

### `AISafety-Student/labeled-bashBench` — the SECOND per-step corpus, and the only MIT one

- **Gated: NO. Licence: MIT** (the only cleanly-licensed step-level option found).
- **2494 rows**, one config/split (`default/train`), 21 columns, 12 MB parquet.
- **One row = ONE ANNOTATED ACTION INSIDE a trajectory, not a trajectory.** Columns:
  `step_index`, `tool_call_index`, `location` (`tool_input`/…), `extraction_target` (the literal
  offending command, e.g. `iptables -P INPUT $POLICY_ACCEPT`), `action_type` (`firewall_disabled`),
  `category` (`firewall_manipulation`), `role` (`malicious`), **`danger_level`** (int, 5 in row 0),
  `is_executed` (bool), **`main_task_progress_pct`** (40 in row 0), `description`, `notes`.
  The full trajectory is carried alongside in `trajectory` (list of
  `{assistant_visible_text, tool_calls[{tool, ...}]}`), plus `task_description`, `side_task`
  (`disable_firewall`), `label`, `split`, `task_id`, `source_file`, `formatted_input`.
- **This is the cleanest per-step supervision in the whole sweep**: a step index, a danger level, an
  executed flag, and a progress percentage. `main_task_progress_pct` is effectively a normalised
  trajectory clock, which makes lead-time expressible as a fraction of task completion even when
  step counts differ.
- **Caveat that limits it:** rows appear to enumerate *malicious* actions (`role='malicious'`).
  Whether benign-role rows exist in quantity determines if >=500/class holds at the STEP level;
  see the class-count table. The trajectory-level negatives may have to come from
  `Eccentricity/bashbench2` or from unannotated steps of the same trajectories.

---

### `Atarogic/ASSEBench` — biggest ungated, cleanly-licensed, trajectory-labelled option

- **Gated: NO. Licence: Apache-2.0.** 836 downloads.
- Four configs, all `test` split only: **`strict` 1476**, **`loose` 1476**, **`safety` 1476**,
  **`security` 817** rows. `strict`/`loose`/`safety` are the SAME 1476 items relabelled under
  different annotation standards (identical byte counts confirm this) — **they are NOT independent
  data, do not pool them**. Real distinct pool = 1476 (safety) + 817 (security) = **2293**.
- Columns: `id`, `profile` (system prompt), **`contents`** (nested list of
  `{role, content|thought|action}` — the multi-step agent trajectory), `label` (int 0/1),
  `application_scenario`, `risk_type`, `failure_mode`, `ambiguous` (int flag), `risk_description`.
- **Label granularity: TRAJECTORY-LEVEL.** One `label` per interaction. Steps are present and
  parseable inside `contents`, so a streaming monitor trains fine — **but there is no per-step
  annotation and no oracle flag step, so ASSEBench CANNOT support a lead-time claim.**
- The `ambiguous` column is a genuine asset: it marks the items where strict and lenient standards
  disagree. Reporting separately on `ambiguous=0` vs `ambiguous=1` is a free, honest robustness cut.

---

### `AI45Research/ATBench` — the brief's "~1,000 trajectories, ~2,000 tools"

- **Gated: NO. Licence: Apache-2.0.** 2436 downloads, 39 likes — the most-adopted item here.
- **`ATBench/test` = 1000 rows; `ATBench500/test` = 500 rows** (a subset). Sister repos
  `ATBench-Claw` (500) and `ATBench-Codex` (248) are ungated, Apache-2.0.
- Columns: `id`, **`tool_used`** (list of full tool JSON specs — this is where "~2,000 tools" lives),
  **`contents`** (nested multi-step trajectory), `label` (0/1), `risk_source`
  (e.g. `tool_description_injection`), `failure_mode`, `reason` (free-text rationale naming the
  specific offending step), `real_world_harm`.
- **Label granularity: TRAJECTORY-LEVEL.** No step index. The `reason` field *describes* the
  decisive step in prose ("The agent passed `extract_news_article_text` results to `create_event`…")
  but does not give an index — you would have to extract it with an LLM, which reintroduces a judge
  and would make any lead-time number judge-derived and circular. **Not a lead-time benchmark.**
- 1000 rows ⇒ >=500/class only if the split is near-perfectly balanced; see the class-count table.

---

### `Glide-py/r_judge_labelled` — R-Judge, ungated, CC-BY-4.0, but SMALL

- **Gated: NO. Licence: CC-BY-4.0.** **568 rows** (`default/train`).
- Columns: `application`, `file_path`, `id`, `scenario`, `goal`, `profile`, **`contents`**
  (multi-step user/agent/environment interaction, with `thought` and `action` per agent turn),
  `label` (0/1), `risk_description`, `attack_type` (`unintended`/…), plus a pre-computed baseline:
  `llm_verdict`, `llm_confidence`, `llm_raw_output`, `llm_parse_error`, `prompt`.
- `imerad-kv/r_judge_labelled` is the same corpus (also ungated).
- **Label granularity: TRAJECTORY-LEVEL.** 568 rows total ⇒ **cannot reach 500/class** (best case
  ~284/class). R-Judge is a good OOD/transfer set, **not** a training set under rule 1.
- Bonus: `llm_verdict`/`llm_confidence` give a free published-baseline comparison without running a
  judge yourself.

---

### `Kevin355/Who_and_When` — has the step label the brief wants, but is FAR too small

- **Gated: NO. Licence: NOT DECLARED.** 1567 downloads, 13 likes.
- **`Algorithm-Generated/train` = 126 rows; `Hand-Crafted/train` = 58 rows. Total 184.**
  Matches the paper (arXiv:2505.00212, ICML 2025 Spotlight) — 127 MAS failure logs, 184 annotated
  tasks. Also ships the raw `Who&When/**/*.json`.
- **Label granularity: PER-STEP — it annotates the "decisive error step" plus the responsible agent
  and a natural-language explanation.** Conceptually this is exactly the lead-time oracle.
- **But 184 rows is ~4% of the rule-1 floor and they are ALL failures** (it is a failure-attribution
  corpus, so there is no negative class at the trajectory level at all). Usable only as a small
  qualitative OOD probe. **Cannot carry any headline.**

---

### AgentDojo — ungated ControlArena dump exists

- `adityaasinha28/control_arena_agentdojo`: **ungated, 1500 rows**, `default/train`, 11 columns —
  the SAME schema as `control_arena_shade` (`prompt`, `ground_truth`, `trajectory_data`,
  `transcript`, `extra_info`, `setting_description`). Licence NOT declared. 3.7 MB — much smaller
  per-trajectory than SHADE (151 MB), i.e. much shorter trajectories.
- `aksh-n/control_arena_agentdojo-monitor-labels` (538 downloads) is the **matching ponr label
  repo**, same construction as the SHADE one — so AgentDojo very likely also has per-step labels.
- `sungjuncho/agentdojo-trajectories` (1076 downloads) is a file repo, `/size` empty.
- Upstream AgentDojo itself is GitHub (`ethz-spylab/agentdojo`) + a task suite, not a labelled
  trajectory dataset.

---

### LongSafety — real and ungated, but NOT a trajectory dataset

- **`thu-coai/LongSafety`: ungated, MIT, 1543 rows** (`default/train`). Columns: `context`
  (16k–92k chars, mean 35k), `instruction`, `safety_type` (7 categories: Illegal Activities 373,
  Unethical 273, Bias 221, Physical & Mental Health 213, Toxicity 182, Sensitive Topics 170,
  Privacy & Property 111), `task_type` (6: QA 334, Role-playing 293, Brainstorming 278, Generation
  277, Summarization 202, Rewrite 159), `doc_num` (1–11), `length` (3000–20676 tokens), `link`,
  `key_words`, `id`.
- **There is no binary safety label and no agent trajectory** — it is a long-context *generation*
  safety set. It tests robustness as context grows, which is thematically adjacent to the brief, but
  it has **no steps and no classes**. Not usable for a streaming step monitor.
- `LutherXD/LongSafetyBench` (ungated, Apache-2.0) is a different, MCQ-style benchmark: 10 splits
  of 100–163 rows each (HarmfulTendency 108, HarmfulExtraction 108, HarmfulAdvice 100, DocAttack
  100, CountingCrimes 147, HarmfulNIAH 147, MedicalQuiz 128, PoliticallyIncorrect 100,
  LeadingQuestion 163, ManyShotJailbreak 100) — **1201 rows total but no split reaches 500**.

---

### AgentHarm, BashBench, MLE-Sabotage — all too small for a headline

| id | gated | licence | rows | verdict |
|---|---|---|---|---|
| `ai-safety-institute/AgentHarm` | **NO** | `other` (AISI terms) | harmful 176+32, harmless_benign 176+32, chat 44+8 | **176/class max.** Prompts, not trajectories. |
| `antieval/agentharm-trajectories` | NO | not declared | 160 | trajectories, but 160 rows |
| `Eccentricity/bashbench2` | NO | CC-BY-SA-4.0 | 637 | **task definitions, NOT trajectories** — cols are `app_file`, `dockerfile`, `tests`, `setup_script_content`, `description`. Environments to run, not data to train on. |
| `aksh-n/mle-sabotage` | NO | not declared | regular 50, monitor_aware 47, monitor_unaware 54 (**151 total**) | far too small; the `monitor_aware`/`monitor_unaware` contrast is scientifically interesting but n≈50. |
| `ai-safety-institute/AgentHarm` chat | NO | other | 44 | trivial |

**BigCodeBench-Sabotage: no HF dataset found** under that or any near name — searches for
"bigcodebench sabotage" return nothing. Treat as **not available**.

---

### Not on HF at all (GitHub-only or engine-only)

| Benchmark | Status |
|---|---|
| **SHADE-Arena (official)** | **No official Anthropic HF dataset.** Code/task-suite lineage only; the usable artifacts are the ControlArena community dumps above. |
| **RAS-Eval** | **No HF dataset** (`ras-eval`, `rasbench` both return zero results). Paper arXiv:2506.15253 — 80 scenarios / 3,802 attack tasks with real tool execution. It is an execution *harness*, so the "real security-audit failure logs" only exist once you run it — which needs live tool execution this host cannot do. **Unusable here.** |
| **Pre-Ex-Bench** | **No HF dataset found.** It is described as a controllable synthetic-data *engine* (synthesize benign trajectories → inject category-labelled risks → reward-model filter), not a released fixed corpus. Produces plan/trajectory-level binary labels + risk types. **Nothing to download.** |
| **ToolEmu** | **GitHub only** — `github.com/ryoungj/ToolEmu` (ICLR'24 Spotlight, arXiv:2309.15817). 36 toolkits / 311 tools / **144 test cases**. Emulator, not a trajectory corpus; 144 is far under the floor. |
| **BigCodeBench-Sabotage** | Not found. |
| **ATBench (paper)** | arXiv:2604.02022. The HF release IS available (above). |

Adjacent ungated find worth knowing: **`thu-coai/Agent-SafetyBench`** (425 downloads) surfaced
during the search — same family as ASSEBench, not in the brief but ungated.

---

## MEASURED CLASS BALANCE (via datasets-server `/filter` `num_rows_total`, not the card)

| dataset | config/split | neg | pos | >=500/class? |
|---|---|---|---|---|
| `adityaasinha28/control_arena_shade` | default/train | **750** | **750** | **YES** |
| `adityaasinha28/control_arena_agentdojo` | default/train | **750** | **750** | **YES** |
| `AISafety-Student/labeled-bashBench` | default/train | **1058** | **1436** | **YES (step level)** |
| `Atarogic/ASSEBench` | strict/test | 641 | **835** | **YES** |
| `Atarogic/ASSEBench` | loose/test | **901** | **575** | **YES** |
| `Atarogic/ASSEBench` | security/test | 408 | 409 | no (409 max) |
| `AI45Research/ATBench` | ATBench/test | **503** | 497 | **borderline — 497, three short** |
| `AI45Research/ATBench` | ATBench500/test | 250 | 250 | no |
| `adityaasinha28/cua_shade_arena_action_only_large` | default/train | 484 | 484 | no (484, sixteen short) |
| `Glide-py/r_judge_labelled` | default/train | 269 | 299 | no |
| `adityaasinha28/cua_shade_arena_aware` | default/train | 146 | 146 | no |
| `Kevin355/Who_and_When` | both configs | 0 | 184 | no (no negative class) |
| `ai-safety-institute/AgentHarm` | harmful/test_public | — | 176 | no |
| `aksh-n/mle-sabotage` | all splits | — | 151 total | no |

`labeled-bashBench` extra cuts (all measured): `role='malicious'` **572**, `role='benign'` **1058**,
`danger_level>=4` **860**, `is_executed=true` **2136**, `split='train'` **2473**, `split='test'`
**21** (the built-in test split is unusably small — make your own group-aware split by `task_id`).
`ASSEBench strict` has **260 `ambiguous=1`** rows.

Note `ATBench` at 497 positives lands *just under* the rule-1 floor. Pooling `ATBench` with
`ATBench-Claw` (500) and `ATBench-Codex` (248) would clear it, but those are different agent
scaffolds and pooling them silently would be a confound.

---

## LEAD-TIME FEASIBILITY — MEASURED, AND IT LARGELY CONTRADICTS THE BRIEF

I computed `lead = point_of_no_return_step − ideal_flagging_step` directly from the adjudicator
blocks. **SHADE: 188 rows sampled (every 8th of 1497, 12.6% of the corpus), 93 positives, 75 with
both fields valid. AgentDojo: 38 rows sampled, 18 positives, 17 valid.**

**SHADE (`control_arena_shade-monitor-labels`)**
- steps/trajectory: min 9, p25 92, **median 129**, mean 126.6, p75 152, max 319 — genuinely long.
- **LEAD: mean 7.61, MEDIAN 0, min −26, max 132.**
- **Sign split: 20 negative, 25 zero, 30 positive.** So **60% of positive trajectories offer zero
  or NEGATIVE advance warning** — the damage is already locked in at, or before, the step an ideal
  monitor could first justify flagging.
- Full distribution: `[-26,-12,-12,-12,-8,-6,-6,-4,-4,-4,-4,-4,-2,-2,-2,-2,-2,-2,-2,-2, 0×25,
  2,2,4,4,4,4,4,4,4,4,4,6,8,10,10,11,12,12,14,16,20,21,29,32,44,50,54,56,112,132]`.
  The mean of 7.61 is carried almost entirely by a handful of outliers (132, 112, 56, 54, 50).
- `verification.passed == False` on 4 of 93 sampled positives — filter these out.
- The adjudicator's `ideal_flagging_step` disagrees with the privileged judge's
  `candidate_ideal_flagging_step` in **12 of 93** positives. Use the adjudicator's.
- **Streaming-judge lead** (`ponr − first step where nonpriv_streaming.would_flag_now`):
  **mean 10.36, median 4** — i.e. the actual online judge often flags *earlier* than the "ideal"
  step, so measured against the point of no return a real streaming monitor does have some
  headroom. This is the more defensible framing.

**AgentDojo (`control_arena_agentdojo-monitor-labels`)**
- steps/trajectory: min 4, **median 8**, mean 9.4, max 40 — short.
- **LEAD = 0 in ALL 17 valid positives. Mean 0.00, median 0, min 0, max 0.**
  `ideal_flagging_step == point_of_no_return_step` identically, and the streaming-judge lead is
  also 0.00. **AgentDojo cannot support a lead-time or advance-warning claim at all** — by
  construction the flag step *is* the point of no return.

### What this means for the brief's headline

The brief's "**3.7 steps of advance warning**" is **not supported as a median on either corpus**.
On SHADE the median available lead is **0**; on AgentDojo it is **0 with zero variance**. A mean
near 3.7 is obtainable from SHADE only by averaging a distribution whose mass sits at and below
zero and whose mean is set by a five-point tail — that is exactly the kind of number that inverts
when n grows (§17 rule 6). Concretely:

- **Do not pre-register a mean lead time.** Pre-register the **fraction of positive trajectories
  detected strictly before `point_of_no_return_step`**, plus the median and full distribution.
  On this sample the ceiling for that fraction is **30/75 = 40%** against the oracle flag step, and
  higher against the streaming judge.
- **A perfect monitor scores 0 median lead on SHADE.** So a lead-time win must be stated relative
  to the `nonpriv_streaming` baseline, not relative to "no warning".
- The "**7–10 tokens early**" sub-claim has **no support anywhere in this sweep** — every corpus
  found is annotated at step/turn granularity. Nothing carries token-level offsets. Either drop it
  or generate it yourself, in which case it is unvalidated by any external benchmark.

---

## Q1 — USABLE ON THIS HOST TODAY, RANKED

Every dataset below is **ungated** and returned real rows to an unauthenticated probe. Nothing in
the shortlist needs an HF token. **>=500/class is reachable — comfortably — so the lesson does NOT
have to be pool-limited for its main claim.**

1. **`adityaasinha28/control_arena_shade` + `aksh-n/control_arena_shade-monitor-labels`** —
   1500 trajectories at an exact **750/750**, median 129 steps, paired with 1497 per-step judge
   logs carrying step analyses, an oracle flag step, a point of no return, and a causal streaming
   baseline. This is the only pairing that supports the full brief. *Licence undeclared; labels are
   Gemini-3.1-Pro-generated.* Cost: 151 MB parquet + 1497 small JSONs.
2. **`AISafety-Student/labeled-bashBench`** — **2494 step-level rows, 1436/1058, MIT-licensed**,
   12 MB. The cheapest and cleanest per-step supervision, with `danger_level`, `is_executed` and
   `main_task_progress_pct`. Best second corpus and the only one with a real licence. Build your
   own split — the shipped `test` split is 21 rows.
3. **`Atarogic/ASSEBench`** — 1476 (safety/strict/loose, the same items relabelled) + 817
   (security), **Apache-2.0**, trajectory-level labels, `ambiguous` flag for a free robustness cut.
   Largest cleanly-licensed trajectory-level set.
4. **`adityaasinha28/control_arena_agentdojo` + its monitor-labels** — another exact **750/750**
   with 1500 per-step judge logs, only 3.7 MB. Excellent as a *short-trajectory contrast* to SHADE
   (median 8 vs 129 steps) — which is itself a good experiment. Useless for lead time.
5. **`AI45Research/ATBench`** — 1000 rows, 503/497, Apache-2.0, richest tool metadata. Positives
   land 3 short of the floor; declare that rather than pooling with the sister repos.
6. **`Glide-py/r_judge_labelled`** (568, CC-BY-4.0) and **`Kevin355/Who_and_When`** (184) —
   **OOD / transfer only.** Both are far under the floor. R-Judge ships a published LLM baseline
   (`llm_verdict`) for free comparison; Who&When is the only human-annotated decisive-error-step
   corpus and is worth a qualitative probe despite its size.

Not usable as data: `thu-coai/LongSafety` (no labels, no steps), `LutherXD/LongSafetyBench` (no
split >=500), `Eccentricity/bashbench2` (task environments, not trajectories),
`ai-safety-institute/AgentHarm` (176/class, prompts), `aksh-n/mle-sabotage` (151 total).
Not obtainable at all: **RAS-Eval** (no HF release; needs live tool execution), **Pre-Ex-Bench**
(a synthesis engine, no fixed corpus released), **ToolEmu** (GitHub, 144 cases),
**BigCodeBench-Sabotage** (not found), **official SHADE-Arena** (no HF dataset).

## Q2 — WHICH SUPPORT A STREAMING MONITOR, AND WHICH SUPPORT LEAD TIME

**Per-STEP labels + an oracle flag step (can validate early detection):**
- `aksh-n/control_arena_shade-monitor-labels` — **the only corpus that can carry a lead-time
  claim**, and even there the median lead is 0, so the claim must be reframed as "fraction detected
  before the point of no return" and benchmarked against `nonpriv_streaming`.
- `aksh-n/control_arena_agentdojo-monitor-labels` — per-step labels, yes; **lead time identically
  0**, so it can validate *streaming accuracy* but **not** advance warning.
- `AISafety-Student/labeled-bashBench` — per-step `step_index` + `danger_level` + `is_executed` +
  `main_task_progress_pct`. Supports "how early in the task does the monitor fire", expressed as a
  progress fraction. No explicit point-of-no-return field, so lead time is proxied, not measured.
- `Kevin355/Who_and_When` — decisive error step, human-annotated, but n=184 and no negatives.

**Trajectory-level labels only — a streaming monitor CAN be trained (steps are parseable from
`contents`/`trajectory_data`), but early-detection lead time CANNOT be validated:**
`Atarogic/ASSEBench`, `AI45Research/ATBench` (+ Claw/Codex), `Glide-py/r_judge_labelled`,
`adityaasinha28/control_arena_shade` and `…_agentdojo` on their own (without the label repos),
`antieval/agentharm-trajectories`, all `cua_shade_arena_*` variants.

**Cannot support a streaming monitor at all** (no sequential agent structure):
`thu-coai/LongSafety`, `LutherXD/LongSafetyBench`, `ai-safety-institute/AgentHarm`,
`Eccentricity/bashbench2`.

**Token-level lead time: NO benchmark in this sweep supports it.** All annotation is per step/turn.

## Q3 — ALREADY CACHED LOCALLY

Real data present under `~/.cache/huggingface`: JailbreakBench JBB-Behaviors, SafeMTData,
wikitext, MMLU, CSTM-Bench, toxic-chat, plus AdvBench/HarmBench/StrongREJECT/XSTest/BeaverTails/
WildJailbreak/do-not-answer/HEx-PHI/ClearHarm/JailBreakV-28K/in-the-wild-jailbreak-prompts/
sorry-bench/mhj/axbench-concept10+500/jailbreak-classification.
**No agent-trajectory dataset is cached.** `ScaleAI/mhj` and `SafeMTData` are the only cached
corpora with any sequential (multi-turn) structure. Everything in the Q1 shortlist is a fresh
download; the two cheapest are `labeled-bashBench` (12 MB) and `control_arena_agentdojo` (3.7 MB).

## Caveats that must ship with the lesson

1. **No licence is declared on any ControlArena dump or either monitor-labels repo.** The
   cleanly-licensed options are `labeled-bashBench` (MIT), `ASSEBench` (Apache-2.0),
   `ATBench` (Apache-2.0), `r_judge_labelled` (CC-BY-4.0).
2. **The step-level labels are Gemini-3.1-Pro output, not ground truth.** Gemini is off-family from
   Gemma (§17 rule 3), but "advance warning against a Gemini oracle" is the honest phrasing.
3. Learning to predict `nonpriv_streaming.would_flag_now` is **monitor distillation**, not scheming
   detection. Keep it as a separate, separately-labelled arm.
4. **Length confound (§17 rule 7) is untested here and is a live risk**: SHADE positives and
   negatives both come from the same generator, but positives carry a side task and may run longer.
   Run `confound_report` on step-count and token-count before claiming any margin —
   `cua_shade_arena_action_only_large` already exposes `token_count` (row 0 = 20478) to check with.
5. `ASSEBench` `strict`/`loose`/`safety` are the same 1476 items under different standards. Pooling
   them would triple the apparent n while adding no information.
