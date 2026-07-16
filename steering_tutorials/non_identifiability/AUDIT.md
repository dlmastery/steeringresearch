# AUDIT — non_identifiability

**Auditor role:** independent paper verifier. Scope: does the cited paper exist,
does the code implement what the lesson claims, are the claims/results honest.
No git, no code/README edits were made.

## Paper existence (the critical check)

| field | finding |
|---|---|
| arXiv id | **2602.06801 — VERIFIED (resolves).** |
| actual title | *On the Non-Identifiability of Steering Vectors in Large Language Models* — matches the README's short title. |
| actual authors | **Sohan Venkatesh, Ashish Mahendran Kurapath** (Manipal Institute of Technology, Bengaluru). |
| venue / dates | arXiv cs.LG; submitted 2026-02-06, latest v4 2026-04-01. |
| method in abstract | Yes — steering vectors are non-identifiable: large equivalence classes of behaviorally indistinguishable interventions; orthogonal perturbations achieve near-equivalent efficacy (Cohen's d < 0.2); mean-diff vs PCA vectors at cos −0.54…+0.32 give near-identical effects. |
| verification | WebFetch of `arxiv.org/abs/2602.06801` + WebSearch confirmed the same title/authors and the abstract's claims. |

## Findings

| check | verdict | evidence |
|---|---|---|
| Paper exists & id resolves | **PASS** | id 2602.06801 resolves; title matches verbatim. |
| Citation attribution correct | **FAIL** | README l.15 and `vectors.py` l.39 credit **"Bereska et al. 2026"**. The real authors are **Venkatesh & Kurapath**. Wrong author name — must be corrected. |
| `[UNVERIFIED]` tag now stale | **CONCERN** | README/code mark the id `[UNVERIFIED]`; it is now verified. The hedge should be removed *together with* the author fix (do not simply un-hedge a wrong citation). |
| Method fidelity — multiple low-cosine directions, matched effect | **PASS** | `vectors.py` builds 6 recipes from one contrast: diff-of-means halves (a,b), PCA-top1 (c), full CAA anchor (d), mean-pooled (e), random-in-PC-span control (f); all unit-normed, sign-aligned to (d); `cosine_matrix` cross-tabs geometry vs `run_nonident` refusal rate at matched alpha. This directly operationalizes the paper's own mean-vs-PCA test and its random-direction control. |
| Fidelity nuance | **CONCERN** | The paper's *headline* construction is explicit null-space / orthogonal perturbations `v + v⊥` (Cohen's d). The lesson relies on naturally-arising low-cosine recipes instead. Faithful in spirit; the strongest test in the paper is not reproduced. Worth a one-line note; not a blocker. |
| Claim accuracy | **PASS** | README frames the result as a screening demo and pre-registers the falsifier (≥2 effective directions with min-cos well below ~0.9). Prediction (cos ~0.3–0.7, comparable refusal) is consistent with the paper's measured −0.54…+0.32. |
| Results honesty (code-only) | **PASS** | Results table marked "pending the GPU run"; screening-tier disclosed (1B model, n=40, one layer/alpha); off-family `Qwen2.5-3B-Instruct` judge documented as recommended; self-judge weakness flagged. |

## Overall verdict

**CONDITIONAL PASS — one required fix.** The paper is real and the method is a
faithful operationalization, but the citation names the **wrong authors**
("Bereska et al." → **Venkatesh & Kurapath**). Correct the author name in
`README.md` and `vectors.py`, then drop `[UNVERIFIED]`. Optionally note that the
paper's primary evidence is orthogonal null-space perturbation, which this lesson
approximates via low-cosine recipes.

*Internal QA pass — independent external review pending (auditor shares a model
family with the author).*
