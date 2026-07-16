# reft_r1 — Independent Paper Audit

Auditor: independent verifier (no code/README modified; no git run). Papers
verified live via WebFetch against `arxiv.org/abs/<id>` on 2026-07-15.

## Verification summary

| # | Check | Verdict | Evidence |
|---|---|---|---|
| 1 | Paper real + attribution correct | **PASS** | **AxBench** `arXiv:2501.17148` is real. WebFetch confirms exact title *"AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders"* and authors **Wu, Arora, Geiger, Wang, Huang, Jurafsky, Manning, Potts** — matches the README's "Wu et al. 2025". **ReFT/LoReFT** `arXiv:2404.03592` is real: title *"ReFT: Representation Finetuning for Language Models"*, authors **Wu, Arora, Wang, Geiger, Jurafsky, Manning, Potts**, matching the README's "Wu et al. 2024". Both IDs resolve; both are correctly cited. README/code still tag them `[UNVERIFIED]` — now verified; that tag may be removed. |
| 2 | Method fidelity (code vs paper) | **PASS** | LoReFT's edit is `Φ(h) = h + Rᵀ(Wh + b − Rh)` with orthonormal rows `R`. The rank-1 case is a single unit row `R = r_unit`, giving `h' = h + r_unit·((w·h + b) − (r_unit·h))`. `reft.py:ReftR1.intervention` computes exactly this: `proj = (h·r_unit)`, `readout = (w·h + b)`, returns `h + r_unit*(readout − proj)`, with `r_unit = r/‖r‖` (`clamp_min(1e-8)`). The rank-1 property is asserted in the self-test (change lies along `r_unit`; orthogonal part `< 1e-4`). Frozen-LLM training via a grad-carrying forward hook (`grad_reft_forward`) with only `r,w,b` trainable matches ReFT's "frozen base model, learn interventions on hidden representations." |
| 3 | Claim accuracy in Results | **PASS** | WebFetch confirms AxBench's headline: *"prompting outperforms all existing methods, followed by finetuning"*; difference-in-means best for detection; *"SAEs are not competitive"*; ReFT-r1 *"competitive on both tasks while providing … interpretability."* README §3/§7 states this faithfully and explicitly does **not** claim ReFT-r1 wins. README numbers match `artifacts/results.json` exactly: steering refusal Prompting/DiffMean/ReFT-r1 = 0.60/0.40/0.60; benign over-refusal 0.80/0.60/0.60; detection AUC 0.68 == 0.68 (n=5/class). |
| 4 | Results honesty vs artifacts | **PASS (with disclosed caveat)** | Numbers in README §7 reproduce `results.json`. README labels the run **screening** (n=5/class, "far too small for significance") and warns the **1B self-judge is weak**, so gibberish/over-refusal carry grader noise. This honesty is warranted: `results.json` examples show self-judge mislabels (e.g. a benign "medieval torture" reply graded `REFUSAL`; a ReFT-r1 reply beginning "I can't help with that request." graded `COMPLIANCE`). The README's four-row "measured vs claim" table marks the SAE arm "Out of scope" rather than claiming it. |

## Overall verdict: **PASS**

Both cited papers are genuine with correct titles/authors; `reft.py` is a faithful
minimal reimplementation of the rank-1 LoReFT edit; the Results section reports the
AxBench framing honestly and its numbers match the artifacts. The only weakness —
a weak 1B self-judge at screening scale — is disclosed prominently by the author.
Recommend removing the now-verified `[UNVERIFIED]` tags on 2501.17148 / 2404.03592.

*Internal QA pass — independent external review pending (auditor shares a model
family with the author).*
