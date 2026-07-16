# AUDIT — gavel

**Auditor role:** independent paper verifier. Scope: does the cited paper exist,
does the code implement what the lesson claims, are the claims/results honest.
No git, no code/README edits were made.

## Paper existence (the critical check)

| field | finding |
|---|---|
| arXiv id (primary) | **2601.19768 — VERIFIED (resolves).** |
| actual title | *GAVEL: Towards Rule-Based Safety Through Activation Monitoring* — matches the README's title verbatim (l.62). |
| actual authors | **Shir Rozenfeld, Rahul Pankajakshan, Itay Zloczower, Eyal Lenga, Gilad Gressel, Yisroel Mirsky** — matches README l.61-62 verbatim, correct order. |
| venue / dates | arXiv; submitted 2026-01-27 (v1), last revised 2026-04-30 (v3). README's "2026" is correct. |
| method in abstract | Yes — activations represented as fine-grained interpretable **cognitive elements** (examples: "making a threat", "payment processing"); composable **predicate rules**; safeguards **reconfigurable without retraining**; transparency/auditability. Matches README's method description (l.67-74) exactly. |
| arXiv id (secondary) | **2310.17389 — VERIFIED (resolves).** *ToxicChat…* (Lin, Wang, Tong, Wang, Guo, Wang, Shang). README cites "Lin et al. 2023" (l.90, l.277) — correct first-author + year + id. |
| verification | WebFetch of both `arxiv.org/abs/` pages confirmed titles, authors, method. |

## Findings

| check | verdict | evidence |
|---|---|---|
| Primary paper exists & id resolves | **PASS** | 2601.19768 resolves; title + method match verbatim. |
| Citation attribution correct (primary) | **PASS** | All six authors named correctly and in order (README l.61-62; `monitor.py` l.27). No fabricated/wrong co-authors — the common bug is absent here. |
| Secondary (ToxicChat) attribution | **PASS** | 2310.17389 resolves; "Lin et al. 2023" is the correct first author/year; used correctly as the dataset source, not overclaimed. |
| Method fidelity — CE library + predicate rule | **PASS** | `monitor.py` implements `CEDetector` (diff-of-means direction + benign-calibrated `tau`), `Rule.any_of/all_of/at_least`, and `GavelMonitor.decide` returning an auditable `{block, scores, fired, triggered_by, reason}`. Directly operationalizes the paper's compositional-CE + reconfigurable-rule thesis. |
| Fidelity nuance (simplification honestly flagged) | **PASS (with disclosed gap)** | Paper's CEs are human-authored fine-grained factors; lesson's CEs are one diff-of-means direction per toxic-chat harm *category*. README l.76-81 and l.248-252 openly call this a "faithful miniature"/"simplification" and note the missing composite ("violence AND payment") example (l.132-134). Honest, not a faithful full reproduction — correctly stated as such. |
| Baseline design | **PASS** | Single broad all-harm direction (`run_gavel.py` l.247-248) is built as the "broad misuse detector" the paper argues past — a fair head-to-head. |
| Results honesty | **PASS** | Results table marked "[PENDING THE GPU RUN]" with no pre-written numbers (README l.220-237); screening-tier disclosed (n≤60/class, one seed, per §7); off-family `Qwen2.5-3B-Instruct` judge documented (l.198, l.244-247); self-judge/coarse-CE/monitor-not-a-fix caveats all stated. `results.json` written before summary print (`run_gavel.py` l.352); no unicode in prints. |

## Overall verdict

**PASS.** Both cited arXiv ids resolve; the primary GAVEL paper is real and its
six authors, title, and method are attributed **correctly and verbatim** (no
fabricated co-authors — the usual failure mode is absent). The code faithfully
operationalizes the compositional-CE + predicate-rule mechanism, and the lesson
is transparent that its per-category CEs are a deliberate simplification of the
paper's fine-grained human-authored elements. Results are honestly marked pending,
screening-tier, off-family-judged. No required fixes. *(Optional nit: README l.63-65
asserts its own WebFetch verification — harmless, and it checks out.)*

*Internal QA pass — independent external review pending (auditor shares a model
family with the author).*
