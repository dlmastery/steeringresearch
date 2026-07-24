# AUDIT — cross_trajectory

**Auditor role:** independent paper/dataset verifier. Scope: do the cited papers
and datasets exist, does the code implement what the lesson claims, are the
claims/results honest. No git, no code/README edits were made. The lead will
WebFetch-verify every arXiv id and HF dataset before merge.

## Paper existence (the critical check)

| cited id | claimed title / role | status |
|---|---|---|
| **arXiv:2606.09084** | *Context-Fractured Decomposition: Distributing Harmful Intent Across Cooperating Agents* — the decomposition-attack thesis | **`[UNVERIFIED]`** — id has a `2606` (2026-06) prefix, later than the auditor's training cutoff; cannot confirm from memory. Lead must WebFetch `arxiv.org/abs/2606.09084`. |
| **arXiv:2604.21131** | *Cross-Session Threats* / source of the CSTM-Bench benchmark | **`[UNVERIFIED]`** — `2604` prefix, post-cutoff; the paired dataset `intrinsec-ai/cstm-bench` must be confirmed to exist and match. |
| **arXiv:2603.13940** | *GroupGuard: Graph-based Detection of Colluding-Agent Attacks* — the `gnn_agg` motivation | **`[UNVERIFIED]`** — `2603` prefix, post-cutoff. Lead must WebFetch. |
| **arXiv:1810.00825** | *Set Transformer* (Lee et al. 2019) — the PMA pooling in `attn_pool` | **PLAUSIBLE — pre-cutoff.** A real, well-known ICML 2019 paper introducing PMA/ISAB; the id and attribution are consistent with the auditor's knowledge. Lead should still confirm the exact id. |
| **arXiv:2602.16935** | *DeepContext* — the turn-level sibling (`multiturn_jailbreak`) | **`[UNVERIFIED]` here, but cross-checked:** used and WebFetch-verified in the `multiturn_jailbreak`/`trajguard` lessons; carried over consistently. |
| **arXiv:2410.10700** | *ActorAttack* (Ren et al. 2024) — source of the Attack_600 positives | **PLAUSIBLE — pre-cutoff.** Matches the id used and verified in the sibling lessons. |

## Dataset existence

| dataset | role | status |
|---|---|---|
| **`SafeMTData/SafeMTData`** `Attack_600` | positives (ActorAttack decompositions) | **PLAUSIBLE — reused from `multiturn_jailbreak`**, where the loader path and `multi_turn_queries`/`query_id` fields were exercised. Lead should confirm the config still loads. |
| **`intrinsec-ai/cstm-bench`** | the real OOD cross-session benchmark | **`[UNVERIFIED]`** — auditor cannot confirm this HF repo exists or that its rows carry `scenario_class` / `sessions_json` as `data.py` assumes. **This is the highest-risk external dependency**; the lead must confirm the repo, splits, and schema before the OOD table is trusted. |

## Findings

| check | verdict | evidence |
|---|---|---|
| Cited ids honestly hedged | **PASS** | Every post-cutoff id (2606.09084, 2604.21131, 2603.13940) is marked `[UNVERIFIED]` in README title block, body, and Repository section — not asserted as fact. |
| Method fidelity — set aggregation | **PASS** | `models.py` implements a permutation-invariant Set-Transformer PMA (`attn_pool`) and a graph message-passing collusion detector (`gnn_agg`) over the set of K trajectory embeddings, with `per_traj_max` as the session-bound baseline — a direct operationalization of "aggregate the K latents to recover the joint intent." |
| Fidelity nuance | **CONCERN** | Positives are **constructed**: an ActorAttack attack's sub-queries are reused *as if* each were a separate agent-trajectory. These are **not** live multi-agent execution traces. Faithful in spirit (the latent structure of a decomposition), but not a reproduction of any deployed multi-agent attack. README Section 10 states this plainly. |
| Leakage control | **PASS** | HARD negatives use disjoint `query_id`s from positives; group-aware `GroupKFold` by `query_id` prevents an attack straddling folds; the confound audit (`kcount_auc`, `totalchar_auc`) checks the trajectory-count/length shortcuts. |
| OOD honesty | **CONCERN** | The OOD table depends entirely on `intrinsec-ai/cstm-bench` existing with the assumed schema (`[UNVERIFIED]`). If the repo is absent or shaped differently, `load_ood_cstm` fails and the OOD table cannot be filled. README already flags CSTM-Bench as small/OOD; the existence risk is the open item. |
| Claim accuracy | **PASS** | README frames the result as a screening demo, marks the table **[PENDING GPU RUN]**, and pre-registers the falsifier (set-aggregators ≤ `per_traj_max` on HARD ⇒ thesis FALSE) before the run, with an explicit no-HARKing clause. |
| Results honesty (code-only) | **PASS** | No numbers are claimed yet; table is `_pending_`; screening-tier, constructed-decomposition, small-OOD, and honest-baseline caveats are all disclosed in Section 10. |

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
