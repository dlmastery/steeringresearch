# AUDIT — curveball

**Auditor role:** independent paper verifier. Scope: does the cited paper exist,
does the code implement what the lesson claims, are the claims/results honest.
No git, no code/README edits were made.

## Paper existence (the critical check)

| field | finding |
|---|---|
| arXiv id | **2603.09313 — VERIFIED (resolves).** |
| actual title | *Curveball Steering: The Right Direction To Steer Isn't Always Linear* — matches README l.29-31 verbatim. |
| actual authors | **Shivam Raval, Hae Jin Song, Linlin Wu, Abir Harrasse, Jeff M. Phillips, Fazl Barez, Amirali Abdullah** — matches README exactly. |
| venue / date | arXiv cs.AI; submitted 2026-03-10. |
| method in abstract | Yes — measures geometric distortion via the **geodesic/Euclidean distance ratio**, finds large concept-dependent distortion, proposes **nonlinear polynomial-kernel-PCA** steering in a feature space beating linear PCA where distortion is strong. Matches what README attributes. |
| verification | WebFetch of `arxiv.org/abs/2603.09313` confirmed title, all 7 authors, date, and the kPCA/geodesic method. |

## Findings

| check | verdict | evidence |
|---|---|---|
| Paper exists & id resolves | **PASS** | id 2603.09313 resolves; title + 7 authors match verbatim. |
| Citation attribution correct | **PASS** | README l.28-31 and `config.py` l.23-27 / `curveball.py` l.29-34 credit the correct authors, title, id, year. No `[UNVERIFIED]` hedge; "Verified" is now accurate. |
| Own-construction honestly labelled | **PASS** | README l.37-42, l.253-257 and both source docstrings state plainly the great-circle geodesic is the **author's own construction "inspired by"** the paper, **not** the paper's kernel-PCA method; "we do not claim to reproduce the paper's numbers." No misrepresentation. |
| Method fidelity — code implements the claim | **PASS** | `curveball_endpoint` (l.90-131) integrates a norm-preserving great-circle rotation with per-step re-aimed tangent; `straight_endpoint` (l.77-87) is the lesson-2 chord; `CurveballContext` shares one hook with a `curved` flag so only geometry differs. CPU self-test (l.369-491) asserts norm-preservation, rotation==alpha, chord inflation, tangent orthogonality, exact hook restoration. |
| Results honesty | **PASS** | Results table (l.208-222) marked **[pending run]**; GPU run disclosed as not yet executed; screening-tier (1B, n=40 eval cap, one layer/alpha) stated; off-family `Qwen2.5-3B-Instruct` judge recommended and 1B self-judge weakness flagged (l.266-268). |
| Negatives kept | **PASS** | Caveats (l.251-273) keep the under-steer / no-gibberish-cut failure modes explicit; matched-budget ≠ matched-behaviour stated; norm shell called only a local manifold proxy. |

## Overall verdict

**PASS.** The paper is real and the id resolves; title and all seven authors
(**Raval, Song, Wu, Harrasse, Phillips, Barez, Abdullah, 2026**) are cited
correctly. The lesson's geodesic-rotation method is its **own construction**,
and the README plus both source docstrings label that distinction honestly —
they do not pass the great-circle arc off as the paper's kernel-PCA method. Code
implements exactly what is claimed, with a strong CPU unit test. Results are
transparently marked pending the GPU run, screening-tier, off-family judged. No
required fixes.

*Internal QA pass — independent external review pending (auditor shares a model
family with the author).*
