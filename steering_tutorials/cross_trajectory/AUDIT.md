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

---

## Addendum 2026-07-28 — the embedder ablation, and what it rules out

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
