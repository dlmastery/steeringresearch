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
| Rubric compliance (>=500/class) | **PARTIAL — confirmed on the real run, and 8 of 21 columns fall short** | `N_PER_CLASS=500`, `N_BENIGN=3000` (raised from 500). The 2026-08-21 run reached **10,065 harmful / 3,000 benign, N=13,065**, so the corpus-level floor is met — but `achieved.requested_vs_achieved` records **8 columns short of 500**: `intellectual_property` 76, `high_risk_gov_decisions` 75, `jailbreak` 106, `malware` 149, `terrorism` 293, `animal_abuse` 357, `child_abuse` 425, `unauthorized_advice` 459. That is a pool ceiling, disclosed per column in the artifact, and it means per-column metrics on those eight rest on small n. `wildguardmix` is **gated** (403) and contributes 0 rows. |
| No LLM judge (detection task) | **PASS** | `results.json` records `"judge": null`; no generation. |
| Claim honesty | **PASS — this row has now been stale twice and is corrected a second time** | **History, kept because the pattern is the point.** It first read "README §10 marks all tables `[PENDING GPU RUN]`" — untrue once the GPU run landed. It was then corrected to MEASURED-THEN-SUSPENDED: §10's tables carried real numbers produced by an encoder that, under transformers 4.55.0 (no reference to `use_bidirectional_attention`), silently ran **causal** instead of bidirectional — proved behaviourally, hidden states bit-identical over a shared 9-token prefix, max abs diff 0.0, versus 5.74 under 5.15.0. **The re-run on the fixed bidirectional encoder landed 2026-08-21 and the suspension is LIFTED.** Current status: §10 reports `artifacts/results_embeddinggemma.json`; the suspended artifact is retained at `results_CAUSAL_ENCODER_SUSPENDED.json` and is **not reversed**. What the re-run settled: the seen-label ordering **reproduced** (trained_head 0.553 > bi_encoder_trained 0.482 > bi_encoder 0.227 > uni_encoder 0.156), so the causal bug was not a sufficient explanation for the lesson's negative conclusions after all — but EXP-G's trained-vs-frozen ordering **inverted** and EXP-H's embedding-geometry claim was **withdrawn**, so it was a sufficient explanation for two specific ones. Claim honesty now hinges on a different item entirely: **falsifier (iv) FIRES** — no method clears the 0.804 TF-IDF content bar — and §10.4 states it. |
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

**PASS on apparatus; the LESSON'S OWN HEADLINE CLAIM does not clear its bar.** All seven
arXiv ids are WebFetch-verified by the lead (titles/authors corrected where they had
drifted), and the EmbeddingGemma-300M model card is the correct backbone reference. The
code is a faithful operationalization of the bi-encoder guardrail *pattern* with honest,
pre-registered falsifiers and clearly disclosed simplifications (frozen general embedder,
cross-encoder-lite baseline, handwritten paraphrases).

**Updated 2026-08-21 — the numbers are no longer pending.** The full run on the fixed
bidirectional encoder is in `artifacts/results_embeddinggemma.json`, and all seven
pre-registered falsifiers are verdicted in README §10.4: **(i) (v) (vi) (vii) survive,
(ii) trips as literally written, (iii) is UNINFORMATIVE** (the frozen baseline is pinned
at exactly 1.000 and the 2026-08-08 amendment fires), and **(iv) FIRES** — no method's
binary harm AUC clears the binding TF-IDF content bar of 0.804 (best is 0.718). By the
lesson's own pre-registration that bars headlining any EXP-A number. The three transfer
arms all ran; the only genuinely external one (`cstm-bench`, n=50) puts every method at or
below chance, so this lesson has **no evidence of external generalisation**.

*Internal QA pass — independent external review pending (auditor shares a model
family with the author).*

---

## Addendum 2026-07-28 — the bi-encoder arm was a degenerate ablation of the cited method

**Defect.** `bi_encoder` scored with raw cosine between **frozen** content and policy
embeddings. Both cited papers **fine-tune** their encoders with InfoNCE + hard negatives:
GLiNER-bi-Encoder ([2602.18487](https://arxiv.org/abs/2602.18487)) states frozen encoders
give only *"baseline performance"* and that task-adapted encoders *"significantly
outperform static frozen representations"*; GLiNER Guard ([2605.05277](https://arxiv.org/abs/2605.05277))
likewise fine-tunes. The lesson's weak numbers were therefore **correctly measured and
wrongly attributed** — an ablation reported as the method.

**Fix.** `ContrastiveBiEncoderGuard`: learned `W_content` / `W_policy` projections,
multi-positive InfoNCE, near-identity init (so the learned space starts at frozen
cosine), trained on **seen policies only**, scoring any policy from its **text**.

| arm | EXP-A seen | EXP-B unseen policy | EXP-E OOD content |
|---|---|---|---|
| frozen cosine | 0.240 | 0.382 | 0.184 |
| **InfoNCE-trained** | **0.575** | **0.294** | **0.397** |

**The finding, and its limit.** Training more than doubles AP under *content* shift and
on seen policies — the papers' claim reproduces. It **degrades** under *policy* shift.
The projection is fitted to **12 policies**, so it learns those twelve directions rather
than a general text-matches-policy relation. The papers buy zero-shot generalisation with
**million-label-scale** training; that scale, not the architecture, is the source of the
property. This is stated in README §10.0 rather than tuned away.

**Not done:** backbone fine-tuning (out of budget), and training at a label scale where
zero-shot transfer could plausibly emerge. Both are the honest blockers, not oversights.

*Internal QA pass — independent external review pending (auditor shares a model family
with the author).*
