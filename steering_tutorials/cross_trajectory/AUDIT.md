# AUDIT — cross_trajectory

> **STATUS 2026-08-08: THE VERDICTS BELOW ARE SUPERSEDED.** Everything from here to
> the 2026-07-28 addendum was written against a **pre-results** version of the
> README and was never re-issued against the shipped one. Its "Results honesty
> (code-only): PASS — no numbers are claimed yet; table is `_pending_`" certifies a
> file that no longer exists; §9 now carries a full results table. An audit that
> certifies a version of a file that no longer exists is the `meerkat` staleness
> defect relocated to the audit layer, and it is recorded here as one.
>
> The current audit of record is **`AUDIT_2026-08.md`** in this folder. Its
> corrections are applied; the four headline items are summarised in
> [§Addendum 2026-08-08](#addendum-2026-08-08--what-the-re-audit-changed) at the
> bottom of this file. Read that first.

**Auditor role:** independent paper/dataset verifier. Scope: do the cited papers
and datasets exist, does the code implement what the lesson claims, are the
claims/results honest. No git, no code/README edits were made. The lead will
WebFetch-verify every arXiv id and HF dataset before merge.

## Paper existence (the critical check)

| cited id | claimed title / role | status |
|---|---|---|
| **arXiv:2606.09084** | *Context-Fractured Decomposition Attacks on Tool-Using LLM Agents: Exploiting Artifact Provenance Gaps* | **VERIFIED 2026-08-20** (lead, WebFetch) — Xiaofeng Lin, Yukai Yang, Daniel Guo, Sahil Arun Nale, Charles Fleming, Guang Cheng; 8 Jun 2026. **NOTE: the title this row previously claimed was WRONG** — it read *"Distributing Harmful Intent Across Cooperating Agents"*, which is this lesson's framing, not the paper's. The paper is about ONE tool-using agent losing artifact provenance ACROSS STEPS. The shipped README already states this fidelity gap correctly; only this audit row was stale. |
| **arXiv:2604.21131** | *Cross-Session Threats in AI Agents: Benchmark, Evaluation, and Algorithms* | **VERIFIED 2026-08-20** (lead, WebFetch) — Ari Azarafrooz, 22 Apr 2026. CSTM-Bench: 26 executable attack taxonomies / 54 scenarios, publicly released on HuggingFace. The paired dataset is separately confirmed BY USE — see the dataset table. |
| **arXiv:2603.13940** | *GroupGuard: A Framework for Modeling and Defending Collusive Attacks in Multi-Agent Systems* | **VERIFIED 2026-08-20** (lead, WebFetch) — Yiling Tao, Xinran Zheng, Shuo Yang, Meiling Tao, Xingjun Wang; 14 Mar 2026. Confirms the README's fidelity note: GroupGuard is **training-free** and combines graph monitoring + honeypot inducement + structural pruning, whereas `gnn_agg` is a TRAINED message-passing classifier with none of those. Inspired-by, not a reproduction. |
| **arXiv:1810.00825** | *Set Transformer* (Lee et al. 2019) — the PMA pooling in `attn_pool` | **PLAUSIBLE — pre-cutoff.** A real, well-known ICML 2019 paper introducing PMA/ISAB; the id and attribution are consistent with the auditor's knowledge. Lead should still confirm the exact id. |
| **arXiv:2602.16935** | *DeepContext* — the turn-level sibling (`multiturn_jailbreak`) | **`[UNVERIFIED]` here, but cross-checked:** used and WebFetch-verified in the `multiturn_jailbreak`/`trajguard` lessons; carried over consistently. |
| **arXiv:2410.10700** | *ActorAttack* (Ren et al. 2024) — source of the Attack_600 positives | **PLAUSIBLE — pre-cutoff.** Matches the id used and verified in the sibling lessons. |

## Dataset existence

| dataset | role | status |
|---|---|---|
| **`SafeMTData/SafeMTData`** `Attack_600` | positives (ActorAttack decompositions) | **PLAUSIBLE — reused from `multiturn_jailbreak`**, where the loader path and `multi_turn_queries`/`query_id` fields were exercised. Lead should confirm the config still loads. |
| **`intrinsec-ai/cstm-bench`** | the real OOD cross-session benchmark | **VERIFIED BY USE — `[SCHEMA UNVERIFIED]` was stale and is dropped 2026-08-20.** The original row said the auditor could not confirm the repo exists or that its rows carry `scenario_class` / `sessions_json` as `data.py` assumes. It does and they do: `artifacts/results.json:ood` carries **measured** AUCs over `intrinsec-ai/cstm-bench` at `n_attack: 52 / n_benign: 56`, so `load_ood_cstm` ran end-to-end against the real schema. Independently corroborated by `multiturn_jailbreak/artifacts/results.json` and `meerkat/artifacts/results.json`, both of which report the same 108 rows split 52/56. The remaining honest caveat is **size, not existence**: 108 rows is a small OOD set and every method sits at or near chance on it (`per_traj_max` 0.5405, `attn_pool` 0.5680, `mean_agg` 0.5000 with a degenerate CI [0.500, 0.500], `gnn_agg` 0.4794 — *below* chance). **This does NOT clear the separate `[SCHEMA UNVERIFIED]` hedge on `tom-gibbs/multi-turn_jailbreak_attack_datasets`** (`data.py:355`, `README.md:408`, `config.py:97`), which was never downloaded on this host and whose loader still probes at runtime — that hedge stands. |

## Findings

| check | verdict | evidence |
|---|---|---|
| Cited ids honestly hedged | **PASS** | Every post-cutoff id (2606.09084, 2604.21131, 2603.13940) is marked `[UNVERIFIED]` in README title block, body, and Repository section — not asserted as fact. |
| Method fidelity — set aggregation | **PASS** | `models.py` implements a permutation-invariant Set-Transformer PMA (`attn_pool`) and a graph message-passing collusion detector (`gnn_agg`) over the set of K trajectory embeddings, with `per_traj_max` as the session-bound baseline — a direct operationalization of "aggregate the K latents to recover the joint intent." |
| Fidelity nuance | **CONCERN** | Positives are **constructed**: an ActorAttack attack's sub-queries are reused *as if* each were a separate agent-trajectory. These are **not** live multi-agent execution traces. Faithful in spirit (the latent structure of a decomposition), but not a reproduction of any deployed multi-agent attack. README Section 10 states this plainly. |
| Leakage control | **PASS** | HARD negatives use disjoint `query_id`s from positives; group-aware `GroupKFold` by `query_id` prevents an attack straddling folds; the confound audit (`kcount_auc`, `totalchar_auc`) checks the trajectory-count/length shortcuts. |
| OOD honesty | **CONCERN — re-based 2026-08-20; the *existence* risk is closed, the *power* risk is not** | The original concern was that `intrinsec-ai/cstm-bench` might not exist or might be shaped differently, leaving the OOD table unfillable. **That risk is retired: the loader ran and the table is filled** (`results.json:ood`, 52 attack / 56 benign). What replaces it is a live concern the old wording hid: at 108 rows nothing separates — `attn_pool` 0.5680 [0.464, 0.673], `per_traj_max` 0.5405 [0.426, 0.648], `gnn_agg` 0.4794 [0.373, 0.589], and `mean_agg` returns exactly 0.5000 with a **degenerate CI [0.500, 0.500]**, which is a sign the scorer is constant on this set rather than merely uninformative. Every CI straddles 0.5. **No OOD transfer claim, positive or negative, is supported by 108 rows** — the arm is reported, not claimed. |
| Claim accuracy | **PASS** — **`[STALE 2026-08-08]`** | README frames the result as a screening demo, marks the table **[PENDING GPU RUN]**, and pre-registers the falsifier (set-aggregators ≤ `per_traj_max` on HARD ⇒ thesis FALSE) before the run, with an explicit no-HARKing clause. **The `[PENDING GPU RUN]` marker is gone from the shipped README** — `artifacts/results.json` (mtime Jul 23 23:11) carries measured AUCs for `easy`, `hard` and `ood`. |
| Results honesty (code-only) | **PASS** — **`[STALE 2026-08-08]`** | No numbers are claimed yet; table is `_pending_`; screening-tier, constructed-decomposition, small-OOD, and honest-baseline caveats are all disclosed in Section 10. **Contradicted by the artifact:** `results.json` claims numbers (hard `mean_agg` 0.9358 vs a 0.7037 `totalchar_auc` bar; OOD `mean_agg` 0.500, CI [0.500, 0.500]) and README §9 carries a full results table. This row certified a pre-results file. |

## Overall verdict

**CONDITIONAL PASS — pending the lead's external verification.** The lesson is an
honest **inspired-by reconstruction** — constructed decompositions plus
permutation-invariant set-aggregators operationalizing the cross-session /
colluding-agent *idea* — **not** a reproduction of any single cited paper, and it
says so. The code-side design (set pooling, group-aware CV, confound audit,
pre-registered falsifier) is sound. Two external dependencies are unverified and
**gate the headline**: (1) the three post-cutoff arXiv ids (2606.09084,
2604.21131, 2603.13940) must WebFetch-resolve to the claimed titles or the
citations must be corrected/removed; (2) the OOD benchmark `intrinsec-ai/cstm-bench`
must exist with the assumed `scenario_class`/`sessions_json` schema or the OOD
table cannot stand. Resolve both, then drop the corresponding `[UNVERIFIED]` tags.

*Internal QA pass — independent external review pending (auditor shares a model
family with the author).*

---

## Addendum 2026-07-28 — the embedder ablation, and what it rules out

> **A causal-encoder suspension was applied here on 2026-08-17 and WITHDRAWN the same
> day.** It assumed `"embedder": "gemma"` meant EmbeddingGemma. It does not:
> `config.py:70` lists `"embeddinggemma"`, `"gemma"` and `"minilm"` as three distinct
> options, and this ablation used `"gemma"` — a Gemma-3-1B **decoder** residual
> (`hidden: 1152`), causal by design and unaffected by the dropped bidirectional flag.
> The file is back at `artifacts/results_gemma_ablation.json`.
>
> **The verdict below still may not be quoted as a live finding** — but for the reason
> that was always on the record, not the encoder bug: the artifact was hand-transcribed
> from a run log and is **not regenerable** from this code (README §9.3). The MiniLM
> column is unaffected.

The headline was measured entirely with **MiniLM**. That left an unpriced alternative
explanation: *"aggregation recovers fractured intent"* might be a property of the
aggregation, or an artifact of that one 384-dim sentence encoder. Re-running with
**Gemma-3-1B layer-12 mean-pooled** (1152-dim decoder residual stream) changes only the
embedder — data, K, splits, n=298/class and the hard-negative construction are identical.

| hard condition | MiniLM | **Gemma** | margin over the 0.704 length bar (Gemma) |
|---|---|---|---|
| `per_traj_max` (baseline) | 0.607 | **0.628** | **−0.076 — BELOW the bar** |
| `mean_agg` | 0.936 | **0.947** | **+0.243** |
| `gnn_agg` | 0.812 | **0.905** | +0.201 |
| `attn_pool` | 0.863 | **0.838** | +0.134 |

**Verdict: the finding HOLDS across embedders.** The per-trajectory collapse and the
aggregator recovery both reproduce on a completely different representation, and the
ordering `mean_agg > gnn_agg > attn_pool > per_traj_max` is preserved. Under *both*
embedders the per-trajectory baseline sits **below** the length-confound bar, so the
claimable result is the aggregation margin and nothing else.

This matters because the alternative was live. M-b in the parent repo had just shown a
single-vector result failing to survive a change of regime, so "it replicates" is not a
formality here — it is the specific thing that could have gone wrong.

**Not run: the OOD arm.** The job was reaped during CSTM-Bench loading, so the Gemma OOD
numbers are absent rather than estimated. The MiniLM OOD result (near chance, 0.48–0.57)
therefore remains the only measured out-of-distribution evidence, and this ablation does
not extend it.

*Internal QA pass — independent external review pending (auditor shares a model family
with the author).*

---

## Addendum 2026-08-08 — what the re-audit changed

The full re-audit is `AUDIT_2026-08.md`. Six of its findings were acted on; this is
the record of what was wrong and what the fix was. **The corrections move against the
lesson, not for it.**

### 1. The `[UNVERIFIED]` gate was declared and then skipped

The section above says "the lead will WebFetch-verify every arXiv id ... before merge."
The lesson shipped with all three tags intact. The verification has now been done — and
it found something worse than an unverified id: **all six ids are real, but four of the
six printed titles were paraphrased or invented**, and none carried authors or a date,
which rubric rule 4 requires explicitly.

| id | what the README printed | the actual paper (verified 2026-08-08) |
|---|---|---|
| 2606.09084 | "Context-Fractured Decomposition: Distributing Harmful Intent Across Cooperating Agents" | **INVENTED subtitle.** *Context-Fractured Decomposition Attacks on Tool-Using LLM Agents: Exploiting Artifact Provenance Gaps* — Lin, Yang, Guo, Nale, Fleming, Cheng; 2026-06-08. About **one** tool-using agent's provenance gaps across steps, **not** K cooperating agents. |
| 2604.21131 | "Cross-Session Threats and the CSTM-Bench Benchmark" | PARAPHRASED (benchmark name spliced in). *Cross-Session Threats in AI Agents: Benchmark, Evaluation, and Algorithms* — Azarafrooz; 2026-04-22. Substance matches. |
| 2603.13940 | "GroupGuard: Graph-based Detection of Colluding-Agent Attacks" | **INVENTED.** *GroupGuard: A Framework for Modeling and Defending Collusive Attacks in Multi-Agent Systems* — Tao, Zheng, Yang, Tao, Wang; 2026-03-14. **Training-free**, graph monitoring + honeypot + pruning. `gnn_agg` is a **trained** classifier with none of it. |
| 1810.00825 | "Set Transformer" | correct short form; full title, six authors and ICML 2019 now printed. |
| 2602.16935 | "DeepContext" | correct short form; full title, authors and date now printed. |
| 2410.10700 | "ActorAttack (Ren et al. 2024)" | attribution right, **title never given**. *LLMs know their vulnerabilities...*; the method is named **ActorBreaker**. |

Tags dropped **after** the titles were corrected, not before. Dropping them first would
have made the file worse: over-hedged and under-verified simultaneously.

### 2. The confound instrument was missing the fold — and the course doc said it was canonical

The local `confound_report` returned the **raw** AUC and never folded `max(auc, 1−auc)`,
while `CONFOUND_DISCIPLINE.md` §4 printed a folded `confound_auc` and declared this very
function "the canonical form." It was not. Live cost: the **easy** condition's
`totalchar_auc = 0.10975` reads *cleaner than chance* and is a **0.890** length tell with
the sign flipped, and README §9.1 printed AUCs of 0.991–0.998 against **no bar at all** —
a **+0.10** margin presented as +0.99. `data.confound_report` is now a thin adapter over
`steering_tutorials/common/confound.py` (four bars, both raw and directionless), and §9.1
prices the easy column.

### 3. The 298/class ceiling was self-imposed

Called "pool-limited" in four places. It was `Attack_600`'s ~596 usable rows cut in half.
`SafeMTData_1K` — **1,680 rows in the same ungated repo** — was never loaded.
`ATTACK_CONFIGS` now takes both. **The honest half:** the 1K config has several actors per
`query_id`, so distinct groups are expected ~500–600, not 1,680; `n` and
`n_distinct_groups` are now separate fields per class. The hard split is also now cut by
**group**, not by row — with the 1K config a row-wise cut would have put the same harmful
goal on both sides.

### 4. The gemma ablation artifact is not evidence as it stands

No code in the repo emits `results_gemma_ablation.json`'s keys, and `RESULTS_PATH` was a
**constant**, so running the gemma arm would have overwritten the MiniLM headline: the
§9.3 side-by-side **could not have been produced by that code**. `RESULTS_PATH` is now
per-embedder; the table carries a "hand-transcribed, not regenerable" banner until a real
run replaces it.

### 5. The embedding cache could return stale vectors against new labels

Keyed on `(condition, embedder)`, validated by **row count alone**. Changing `CT_SEED`
reshuffles the hard half-split at identical count. Now handled by `embed_ct.py`: a
SHA-256 of the texts plus seed/K/n/dim/embedder is stored **in the npz and in the
filename** and **asserted** on load, raising rather than silently rebuilding. A CPU
self-test proves the guard fires. **Cross-lesson note:** the same count-only validation
still lives in `multiturn_jailbreak/embed.py:114-119` and was **not** edited from here.

### 6. The EmbeddingGemma mandate was presented as a free choice

CLAUDE.md §17 and the standing user instruction mandate **`google/embeddinggemma-300m`**.
The headline is MiniLM; the "gemma" arm is a Gemma-3-1B **decoder** residual. Neither is
EmbeddingGemma. The old justification (§18.5: "gated, this host has no HF token") is
**stale for the model** — the weights are on disk at `models/google/embeddinggemma-300m`
(1.2 GB safetensors, verified). So both shipped arms are **non-compliant substitutes**,
not necessities. `CT_EMBED=embeddinggemma` is now wired and is `[PENDING RUN]`.

### Also corrected

- `data.py`'s docstring claimed the hard classes were "length/count-matched". Count yes
  (0.500 by construction); **length no** — its own `confound_report` measured 0.704.
- The falsifier was registered only against `per_traj_max`; restated against the binding
  confound bar, under which `per_traj_max` **already fails** on hard, both embedders.
- `mean_agg` was described as "mean-pool the K embeddings"; it pools `[mean, max, std]`.
  It is the **top** method under both embedders, so the winner's description was wrong.
- The OOD arm kept the **5 longest of ~26** sessions per scenario on a `dilution` split;
  and its predictions were **degenerate** (all-positive; `mean_agg` constant at AUC 0.500,
  CI [0.500,0.500]), not merely near chance. Selection default is now `uniform`, features
  are clipped, degeneracy is logged — and the OOD null is caveated as not-yet-supported
  **before** it is called a negative result.

**Open / not done here:** no run was executed. Every §9 number remains the old
`Attack_600`-only, 298/class, `longest`-5-sessions MiniLM run. The enlarged pool, the
mandated encoder, the corrected OOD selection and the gemma re-run are all `[PENDING RUN]`.

*Internal QA pass — independent external review pending (auditor and implementer share a
model family). No git command was run and no model was loaded while applying these fixes;
`arXiv:2409.00137` was WebFetch-verified during this pass, the other six ids by the
`AUDIT_2026-08.md` auditor.*
