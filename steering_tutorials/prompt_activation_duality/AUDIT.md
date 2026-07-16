# AUDIT — prompt_activation_duality

**Auditor role:** independent paper verifier. Scope: does the cited paper exist,
does the code implement what the lesson claims, are the claims/results honest.
No git, no code/README edits were made.

## Paper existence (the critical check)

| field | finding |
|---|---|
| arXiv id | **2605.10664 — VERIFIED (resolves).** |
| actual title | *Prompt-Activation Duality: Improving Activation Steering via Attention-Level Interventions* — matches the README title verbatim. |
| actual authors | **Diancheng Kang, Zheyuan Liu, Ningshan Ma, Yue Huang, Zhaoxuan Tan, Meng Jiang** — matches README l.87–88 and `duality.py` l.23–25 exactly. |
| venue / date | arXiv cs.CL; submitted 2026-05-11. |
| method in abstract | Yes — introduces **GCAD** (Gated Cropped Attention-Delta): extracts steering from system-prompt contributions to self-attention, applies with token-level gating; concludes steering is more reliable when interventions follow prompt-mediated pathways. Matches the "duality" the README describes. |
| verification | WebFetch of `arxiv.org/abs/2605.10664` confirmed title, all six authors, GCAD, and the duality claim. |

## Findings

| check | verdict | evidence |
|---|---|---|
| Paper exists & id resolves | **PASS** | 2605.10664 resolves; title matches verbatim. |
| Citation attribution correct | **PASS** | All six authors named correctly in README (l.87) and `duality.py` (l.23); no fabricated co-authors; no stale `[UNVERIFIED]` tag. |
| Method fidelity | **PASS (honestly simplified)** | Half 1 `prompt_shift_direction`/`cosine`/`random_cosine_baseline` (duality.py l.39–96) measure prompt-shift vs diff-of-means alignment; Half 2 `AttentionSteeringContext` (l.115) hooks `self_attn` output with relative-add + special-token guard + exact restore, verified by the CPU self-test (l.301–384). README l.92–96 and caveats l.291–295 openly state GCAD's token-gated cropping + multi-turn KV fix are out of scope — "faithful, simplified construction," not a full reproduction. |
| Results honesty | **PASS** | Results table marked **[PENDING GPU RUN]** (l.217); no numbers invented; screening tier disclosed (single seed, 1B model, n=20/arm, one layer/alpha; l.239–244); off-family **Qwen2.5-3B-Instruct** judge documented (l.61, l.258); abliterated-base + weak-judge limits stated as prominently as the claims (l.296–298). |

## Overall verdict

**PASS.** The paper is real, the id resolves, and the citation attributes all six
authors and the GCAD/duality claim correctly. The code faithfully operationalizes
the two core ideas (prompt-shift~vector alignment; attention-output injection) and
is honest that the paper's token-gated cropping and multi-turn KV fix are
deliberately out of scope. Results are unfilled pending the GPU run, screening tier
and off-family judge are disclosed, and no negatives are hidden. No required fixes.

*Internal QA pass — independent external review pending (auditor shares a model
family with the author).*
