# AUDIT — fine_grained

**Auditor role:** independent paper verifier. Scope: does the cited paper exist,
does the code implement what the lesson claims, are the claims/results honest.
No git, no code/README edits were made.

## Paper existence (the critical check)

| field | finding |
|---|---|
| arXiv id | **2602.04428 — VERIFIED (resolves).** |
| actual title | *Fine-Grained Activation Steering: Steering Less, Achieving More* — matches the README verbatim. |
| actual authors | Zijian Feng, Tianjiao Li, Zixiao Zhu, Hanzhang Zhou, Junlang Qian, Li Zhang, Jia Jim Deryl Chua, Lee Onn Mak, Gee Wah Ng, Kezhi Mao. |
| venue / dates | **ICLR 2026** (published); arXiv 2026-02-04; code at github.com/zijian678/AUSteer. |
| method in abstract | Yes — decomposes block activations into atomic units (AUs), uses an **activation-momentum** metric on contrastive samples to find the most discriminative AUs, then applies **adaptive per-input steering strengths** to those AUs ("AUSteer"). "Steering less, achieving more" with ≤100 activations. |
| verification | WebFetch + WebSearch confirmed title/id/venue and the AUSteer method; GitHub repo exists. |

## Findings

| check | verdict | evidence |
|---|---|---|
| Paper exists & id resolves | **PASS** | id 2602.04428 resolves; title matches; ICLR 2026; public code. |
| Method fidelity — sparse steering | **CONCERN** | `sparse.py:sparsify` keeps the **top-k% coordinates by |magnitude| of the diff-of-means vector**, matched norm, single global alpha. The paper's AUSteer selects units by an **activation-momentum discriminativeness** metric (not raw magnitude) and uses **adaptive per-input strengths**. The lesson captures the headline ("steer fewer coordinates, comparable/less collateral") but **not the paper's actual selection rule or adaptive strengths**. |
| Claim accuracy | **CONCERN** | README l.10–13 & 128 present top-k magnitude sparsification as "the paper's claim." It is a *simplified reconstruction inspired by* the paper, not AUSteer. Recommend relabeling: keep the (now-verified) citation, but state plainly that this lesson implements magnitude-based sparsification, whereas AUSteer uses activation-momentum AU selection + adaptive strengths. |
| `[UNVERIFIED]` tag now stale | **CONCERN** | id is verified; the `[UNVERIFIED]` markers (README l.11, `sparse.py` l.13) should be removed, paired with the fidelity-scope clarification above. |
| Implementation correctness | **PASS** | `sparsify` is clean and unit-tested: exact top-k support via `argpartition`, matched-norm renormalization, `keep_frac=1.0` identity passthrough, guard-rail on bad fractions. `SparseSteeringContext` delegates hook mechanics to lesson 2 unchanged. |
| Results honesty (code-only) | **PASS** | Results row marked "Pending GPU run"; screening-tier disclosed (N_EVAL=60/class, single seed, cannot clear the rigor contract); null (`best_sparse=None`) promised as prominently as a win; off-family `Qwen2.5-3B-Instruct` judge documented. |

## Overall verdict

**PASS with required labeling fix.** The paper is real, correctly titled, and
correctly cited (authors/venue fine). The **method the lesson implements
(top-k-magnitude sparsification) is not AUSteer's actual mechanism**
(activation-momentum AU selection + adaptive strengths). Keep the citation but
relabel the lesson as a simplified, magnitude-based reconstruction *inspired by*
the paper, and drop `[UNVERIFIED]`. No code bug found.

*Internal QA pass — independent external review pending (auditor shares a model
family with the author).*
