# AUDIT — decomposing_prompting

**Auditor role:** independent paper verifier. Scope: does the cited paper exist,
does the code implement what the lesson claims, are the claims/results honest.
No git, no code/README edits were made.

## Paper existence (the critical check)

| field | finding |
|---|---|
| arXiv id | **2606.03093 — VERIFIED (resolves).** |
| actual title | *Decomposing how prompting steers behavior* — matches the README verbatim. |
| actual authors | **Fan L. Cheng and Nikolaus Kriegeskorte** — matches README §1/§8 exactly. |
| venue / date | arXiv; submitted 2026-06-02. |
| method in abstract | Yes — a nested geometric decomposition (translation → rigid+uniform-scale → sequential axis-scaling → affine → nonlinear) of prompt-induced representational change across LLMs/VLMs. Confirmed via WebFetch of `arxiv.org/abs/2606.03093`. |

## Findings

| check | verdict | evidence |
|---|---|---|
| Primary paper exists & id resolves | **PASS** | 2606.03093 resolves; title + both authors match README verbatim; already tagged **Verified**, not `[UNVERIFIED]`. |
| Secondary citations resolve & attributed correctly | **PASS** | All resolve with correct first authors: 2501.17148 AxBench (**Wu** et al.); 2312.06681 CAA (**Nina Panickssery**, née Rimsky — README's "Panickssery (Rimsky)" is correct); 2406.11717 refusal (**Arditi** et al.); 2310.17389 ToxicChat (**Lin**); 2404.01318 JailbreakBench (**Chao**). No fabricated authors. |
| Method fidelity | **PASS (own construction, faithful in spirit)** | `decompose.py` splits per-prompt delta `d(x)=act(WITH instr)−act(WITHOUT)` into on-direction `<d,u>u` + orthogonal residual; reports on-direction energy fraction, pairwise-cosine consistency, shared-translation fraction; pure-NumPy CPU self-test passes. Faithfully reproduces only the paper's **translation tier**. |
| Scope honesty | **PASS** | §7 states plainly this is not the paper's multi-model, multi-tier (translation→rigid→affine→nonlinear) fit — the affine/nonlinear tiers are left as extension. |
| Emphasis nuance | **CONCERN (minor)** | Paper's headline is that the **affine** (cross-dimensional mixing) tier is the key mechanism; README §1 leans on translation dominance. Lesson is upfront it reproduces only the translation tier and pre-registers a falsifier, so not misleading — one clarifying line optional. |
| Results honesty | **PASS** | Results table carries a `[PENDING RUN]` banner; predictions stated, "measured" column blank; screening-tier, off-family `Qwen2.5-3B-Instruct` judge, and 1B self-judge weakness all disclosed; falsifier pre-registered. |

## Overall verdict

**PASS.** The primary paper (arXiv:2606.03093, *Decomposing how prompting steers
behavior*, Cheng & Kriegeskorte, 2026) and all five secondary citations resolve
with correct titles and first authors — no fabricated attributions. The code is a
faithful (own-construction) reproduction of the paper's translation tier, honestly
scoped, with results marked pending and screening-tier caveats disclosed. Optional
only: one line noting the paper's headline mechanism is the affine tier, which this
lesson does not fit.

*Internal QA pass — independent external review pending (auditor shares a model
family with the author).*
