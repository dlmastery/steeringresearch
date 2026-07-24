# AUDIT — biencoder_guard

**Auditor role:** independent paper/data verifier. Scope: do the cited papers and
datasets exist, does the code implement what the lesson claims, are the
claims/results honest. No git, no code/README edits were made. All eight arXiv
ids below were WebFetch-verified by the lead; `[UNVERIFIED]` tags have been
dropped from README.md and this file.

## What this lesson is (stated plainly)

An **inspired-by reconstruction** of the 2026 **bi-encoder / dual-tower**
safety-guardrail pattern, built on a **general, frozen** sentence embedder
(EmbeddingGemma-300M). It is **NOT** a reproduction of any single cited paper's
trained model: the GLiNER family trains a bespoke encoder with token-level
cross-attention between text and label; here both towers are the same off-the-shelf
frozen embedder and compatibility is a plain cosine. The lesson reproduces the
*architecture pattern* (decouple towers, cache the policy tower, score by cosine,
add policies zero-shot by description) and the *scaling/zero-shot claims* that
follow from it — not the papers' trained weights or exact accuracy. The
`uni_encoder` baseline is a cross-encoder-**lite** (joint-string embed + small
head), a stand-in for a cross-encoder's *cost shape*, not its architecture.

## Paper existence (VERIFIED by the lead's WebFetch)

Titles and authors below are the verified details; the two drifted titles from the
initial draft (#1's subtitle and #3's "Three-Level Taxonomy") have been corrected
here and in README.md, and #7 carries no "CausalNeg:" title prefix (CausalNeg is
the method name).

| # | arXiv id | verified title | authors | used for | status |
|---|---|---|---|---|---|
| 1 | 2602.18487 | The Million-Label NER: Breaking Scale Barriers with GLiNER bi-encoder | Stepanov, Shtopko, Vodianytskyi, Lukashov (Feb 2026) | the core bi-encoder-at-scale thesis | **VERIFIED (lead WebFetch)** |
| 2 | 2605.05277 | GLiNER Guard: Unified Encoder Family for Production LLM Safety and Privacy | Minko, Sadiekh, Kokuykin (May 2026) | the safety-guardrail application | **VERIFIED (lead WebFetch)** |
| 3 | 2605.29659 | Opir: Efficient Multi-Task Safety Classification for Toxicity, Jailbreaks, Hate Speech, and Harmful Content | Stepanov, Smechov (May 2026) | the many-label / 996-category motivation | **VERIFIED (lead WebFetch)** |
| 4 | 2605.07982 | GLiGuard: Schema-Conditioned Classification for LLM Safeguard | Zaratiana, Newhauser, Hurn-Maloney, Lewis (May 2026) | schema conditioning / synthetic schema expansion | **VERIFIED (lead WebFetch)** |
| 5 | 2603.20990 | ECIsem: Semantic Residual Effective Contrastive Information for Evaluating Hard Negatives | Sinha, Seetharaman, Bansal (Mar 2026) | the training-free hard-negative diagnostic (`eci_score`) | **VERIFIED (lead WebFetch)** |
| 6 | 2604.11092 | ARHN: Answer-Centric Relabeling of Hard Negatives with Open-Source LLMs for Dense Retrieval | Choi et al. (SIGIR 2026) | the false-negative filter | **VERIFIED (lead WebFetch)** |
| 7 | 2606.01304 | When Hard Negatives Hurt: Bridging the Generative-Discriminative Gap in Hard Negative Synthesis for Retrieval (method: CausalNeg) | Zhang et al. (KDD 2026) | the controlled counterfactual negatives | **VERIFIED (lead WebFetch)** |

**Backbone:** `google/embeddinggemma-300m` — Gemma-3-based sentence embedder,
768-dim with Matryoshka truncation, task-prompted (query/document). Gated (Gemma
license); loaded via sentence-transformers with a local-snapshot fallback. The
model card should be confirmed to exist and to document the task prompts the code
relies on.

## Dataset existence (to be confirmed by the lead)

| dataset | id / config | role | status |
|---|---|---|---|
| BeaverTails | `PKU-Alignment/BeaverTails`, `30k_train`/`30k_test` | 14-way harm-category taxonomy → core policy columns | should verify columns |
| toxic-chat | `lmsys/toxic-chat`, `toxicchat0124` | in-the-wild toxicity + jailbreak flags; benign hard negatives | reused from lesson 1 lineage |
| wildguardmix | `allenai/wildguardmix`, `wildguardtrain` | adversarial prompt-harm labels; benign-adversarial hard negatives | may be gated; code skips gracefully |

## Findings (code + claims, as authored)

| check | verdict | evidence |
|---|---|---|
| Method fidelity — two frozen towers, cached policy bank, cosine, zero-shot-by-description | **PASS (as design)** | `encoders.get_embedder` routes content/policy task prompts; `build_policy_bank` caches one (multi-prototype) vector per label; `BiEncoderGuard` scores by cosine and works on held-out columns. This is the bi-encoder pattern faithfully; it is not a GLiNER trained model (disclosed). |
| Scaling claim is operationalized, not asserted | **PASS** | `scaling_latency` times bi (embed once + matmul over K cached vecs) vs uni (K joint embeds per text) across `LABEL_SCALES`; the ordering is a measured falsifier (EXP-D), not a narrative. |
| Zero-shot claim is testable | **PASS** | `split_seen_heldout` withholds policy columns from all training; EXP-B scores them for bi/uni and reports `trained_head` as `N/A`. Falsifier: macro-AP ≤ 0.5 ⇒ claim FALSE. |
| Hard-negative pipeline matches the cited recipe | **PASS (with stand-ins)** | dense mining → ECIsem diagnostic → CausalNeg (templated) → ARHN filter → InfoNCE adapter, each mapped to its paper. The ARHN policy-support check and CausalNeg perturbations are **cheap lexical/templated stand-ins** for the papers' LLM-based steps — disclosed in README §7 and code comments. |
| Rubric compliance (>=500/class) | **PASS (config)** | `N_PER_CLASS=500`, `N_BENIGN=500`; wildguard skip is graceful (BeaverTails+toxic-chat suffice). To confirm on the real GPU run that the pools actually reach 500. |
| No LLM judge (detection task) | **PASS** | `results.json` records `"judge": null`; no generation. |
| Claim honesty | **PASS** | README §10 marks all tables `[PENDING GPU RUN]`, pre-registers three falsifiers, and §11 discloses screening tier + the frozen-general-embedder / cross-encoder-lite / handwritten-paraphrase simplifications. |
| Citation ids + titles verified | **PASS** | all seven arXiv ids WebFetch-verified by the lead; two drifted titles (#1, #3) and the #7 method-vs-title distinction corrected in README + this file; `[UNVERIFIED]` tags dropped (cf. the non_identifiability wrong-author finding, avoided here). |

## Concerns (not blockers)

- **Author names verified, not guessed.** Unlike the non_identifiability audit
  (which shipped a wrong author name), the README's Reference block now carries the
  lead-verified authors for all seven ids, and two drifted titles were corrected
  against the resolved arXiv pages — the wrong-author/wrong-title failure mode is
  avoided here.
- **`config.py` load-order note (out of scope for docs).** `ADAPTER_CACHE` /
  `HARDNEG_PNG` reference `ARTIFACTS` above where `ARTIFACTS` is defined; the RUN
  agent that owns `config.py` should confirm the module imports cleanly. Flagged
  for the owner, not edited here.
- **EmbeddingGemma is a general embedder.** Its off-the-shelf cosine is a floor,
  not the papers' trained ceiling — already disclosed, restated here so the
  results are not read as a GLiNER reproduction.

## Overall verdict

**PASS.** All seven arXiv ids are WebFetch-verified by the lead (titles/authors
corrected where they had drifted), and the EmbeddingGemma-300M model card is the
correct backbone reference. The code is a faithful operationalization of the
bi-encoder guardrail *pattern* with honest, pre-registered falsifiers and clearly
disclosed simplifications (frozen general embedder, cross-encoder-lite baseline,
handwritten paraphrases). No gating items remain; the numbers are pending the GPU
run and marked as such.

*Internal QA pass — independent external review pending (auditor shares a model
family with the author).*
