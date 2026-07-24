# AUDIT — meerkat

**Auditor role:** independent paper verifier. Scope: do the cited papers exist,
does the code implement what the lesson claims, are the claims/results honest.
No git, no code/README edits were made.

## Paper existence (the critical check)

| field | finding |
|---|---|
| arXiv id | **2604.11806 — VERIFIED (resolves).** |
| actual title | *Detecting Safety Violations Across Many Agent Traces* — the README's "Meerkat" short name matches the system named in the paper. |
| actual authors | **Adam Stein, Davis Brown, Hamed Hassani, Mayur Naik, Eric Wong** — matches `config.py`'s "Stein, Brown, Hassani, Naik & Wong" verbatim. |
| venue / dates | arXiv cs.AI / cs.CL; submitted 2026-04-13; 35 pages, 17 figures. |
| method in abstract | Yes — Meerkat combines **clustering with agentic search** to detect safety violations (misuse, misalignment, task gaming) across large collections of agent traces; finds "sparse failures without relying on seed scenarios, fixed workflows, or exhaustive enumeration," beating baseline monitors. |
| verification | WebFetch of `arxiv.org/abs/2604.11806` confirmed the title, all five authors, and the abstract's clustering-plus-agentic-search claim. |

## Secondary citations

| id | finding |
|---|---|
| **2604.21131** (CSTM-Bench, OOD) | **VERIFIED.** Actual title *Cross-Session Threats in AI Agents: Benchmark, Evaluation, and Algorithms*, author **Ari Azarafrooz**; submitted 2026-04-22; introduces CSTM-Bench for cross-session attacks. Matches the README's use as the OOD benchmark. |
| **2603.13940** (GroupGuard) | **VERIFIED.** Actual title *GroupGuard: A Framework for Modeling and Defending Collusive Attacks in Multi-Agent Systems*, authors **Yiling Tao, Xinran Zheng, Shuo Yang, Meiling Tao, Xingjun Wang**; a training-free graph-based collusion defense. Cited only as collusion-detection context; consistent. |
| **2410.10700** (ActorAttack, positives) | Reused across the series as the SafeMTData `Attack_600` source (Derail Yourself / ActorAttack). Treated as VERIFIED by the sibling lessons' audits. |
| **2309.07597** (BGE / C-Pack, embedder) | Marked **`[UNVERIFIED]`** in the README — the id for the bge-base-en-v1.5 embedder paper was not independently WebFetch-confirmed in this pass. The embedder itself (`BAAI/bge-base-en-v1.5`) is the real Hub model the paper uses; only the paper id carries the hedge. |

## Findings

| check | verdict | evidence |
|---|---|---|
| Primary paper exists & id resolves | **PASS** | 2604.11806 resolves; title + all five authors confirmed. |
| Citation attribution correct | **PASS** | `config.py` credits "Stein, Brown, Hassani, Naik & Wong" — exactly the paper's authors. No wrong-author error (contrast the `non_identifiability` finding). |
| OOD + collusion citations | **PASS** | 2604.21131 and 2603.13940 both resolve with matching topics. |
| Method fidelity — clustering surfaces a sparse campaign | **PASS** | `cluster.py` embeds traces with bge, `choose_k` by silhouette, k-means the repository, and `KMeansEnrich` scores each trace by its cluster's seed-estimated violation fraction — the paper's clustering intuition. Benchmarked at the paper's ~5% sparse base rate with AP as the headline metric. |
| Fidelity nuance — the agentic-search step | **CONCERN (disclosed)** | This is an **inspired-by reconstruction**, NOT the paper's system. Meerkat pairs clustering with an **agentic-search** component that reasons over clusters and describes violations in natural language; the lesson implements only a **cluster-density localizer** (k-means + enrichment) and is **semi-supervised** (a small seed) where the paper is closer to unsupervised. The README states this plainly in Section 10; not a blocker, but the strongest part of the paper is not reproduced. |
| Claim accuracy | **PASS** | README frames the result as a screening demo and pre-registers the falsifier (if `kmeans_enrich` AP ≤ `per_trace` AP at the 5% sparse rate, the thesis is FALSE). The prediction (clustering beats per-trace on sparse AP) matches the paper's headline. |
| Results honesty (code-only) | **PASS** | Results table marked **[PENDING GPU RUN]**; screening-tier disclosed (one embedder, one seed fraction, a few hundred traces); no LLM judge (detection lesson) documented; length confound (`length_auc`) audited and quoted beside the APs; adaptive-dispersion weakness flagged as the paper's own. |

## Overall verdict

**PASS — inspired-by reconstruction, honestly scoped.** All primary and secondary
arXiv ids resolve with correct titles and authors; the author attribution is
correct (no fix required). The code faithfully operationalizes Meerkat's
**clustering** intuition (bge + k-means + cluster-enrichment) at the paper's sparse
~5% regime, but it is **not** the paper's full agentic-search system and is
semi-supervised — both stated plainly in the README's caveats. The only residual
item is the `[UNVERIFIED]` tag on the BGE/C-Pack id (2309.07597), which should be
confirmed or the hedge kept.

*Internal QA pass — independent external review pending (auditor shares a model
family with the author).*
