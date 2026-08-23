# Bi-Encoder Guardrail — cache the policy tower once, moderate a million labels for a cosine

> **Reference:** [The Million-Label NER: Breaking Scale Barriers with GLiNER bi-encoder (arXiv:2602.18487)](https://arxiv.org/abs/2602.18487) — Stepanov, Shtopko, Vodianytskyi, Lukashov (Feb 2026); [GLiNER Guard: Unified Encoder Family for Production LLM Safety and Privacy (arXiv:2605.05277)](https://arxiv.org/abs/2605.05277) — Minko, Sadiekh, Kokuykin (May 2026); [Opir: Efficient Multi-Task Safety Classification for Toxicity, Jailbreaks, Hate Speech, and Harmful Content (arXiv:2605.29659)](https://arxiv.org/abs/2605.29659) — Stepanov, Smechov (May 2026); [GLiGuard: Schema-Conditioned Classification for LLM Safeguard (arXiv:2605.07982)](https://arxiv.org/abs/2605.07982) — Zaratiana, Newhauser, Hurn-Maloney, Lewis (May 2026); the shared backbone [EmbeddingGemma-300M (model card)](https://huggingface.co/google/embeddinggemma-300m). Hard-negative data-synthesis line: [ECIsem: Semantic Residual Effective Contrastive Information for Evaluating Hard Negatives (arXiv:2603.20990)](https://arxiv.org/abs/2603.20990) — Sinha, Seetharaman, Bansal (Mar 2026); [ARHN: Answer-Centric Relabeling of Hard Negatives with Open-Source LLMs for Dense Retrieval (arXiv:2604.11092)](https://arxiv.org/abs/2604.11092) — Choi et al. (SIGIR 2026); [When Hard Negatives Hurt: Bridging the Generative-Discriminative Gap in Hard Negative Synthesis for Retrieval (arXiv:2606.01304)](https://arxiv.org/abs/2606.01304) — Zhang et al. (KDD 2026); method name **CausalNeg**.

> ## RE-MEASURED 2026-08-21 on the FIXED bidirectional encoder — the suspension is lifted
>
> **What was wrong (2026-08-17).** `google/embeddinggemma-300m` is a **bidirectional**
> encoder, but transformers 4.55.0 contains zero references to
> `use_bidirectional_attention`, so the flag was silently dropped and **both towers ran
> causal**. Proved behaviourally: bit-identical hidden states (`0.000000e+00`) over a
> 9-token shared prefix. Full write-up:
> [`audits/AUDIT_2026-08-17_embeddinggemma_causal.md`](../../audits/AUDIT_2026-08-17_embeddinggemma_causal.md).
>
> - **The suspended artifact is retained** at
>   `artifacts/results_CAUSAL_ENCODER_SUSPENDED.json`. It is **not deleted and not
>   reversed** — it is the record of what a crippled encoder produced.
> - **The current artifact is `artifacts/results_embeddinggemma.json`**, written by the
>   2026-08-21 re-run on the fixed bidirectional encoder at `N_PER_CLASS=500`,
>   `N_BENIGN=3000`, 21 policy columns, corpus **N = 13,065**. Every number in §10 below
>   is read from it.
> - **The seen-label ORDERING reproduces**, so the causal-encoder run's central
>   conclusion was **not** an artifact of the bug:
>   `trained_head` **0.553** > `bi_encoder_trained` **0.482** > `bi_encoder` **0.227**
>   > `uni_encoder` **0.156** (macro-AP, seen labels). The suspended run had the same
>   order. What the fix changed is magnitudes and the transfer picture, not the ranking.
> - **Held-out zero-shot flips the order**, as it did before: `bi_encoder` **0.385**
>   beats `bi_encoder_trained` **0.178**, and `trained_head` is **N/A** — it has no
>   column for a policy it never trained on and cannot score it at all. That structural
>   fact never depended on the attention mode.
> - **One thing got WORSE and it is the load-bearing caveat:** on the *test split* the
>   binding confound bar is the **TF-IDF content bar at 0.804**, and **no method clears
>   it** (best is `trained_head` at 0.718, margin **−0.086**). Pre-registered falsifier
>   **(iv) FIRES** — see §10.4. Read every accuracy number in this lesson against that.

> Earlier safety lessons catch an attack **inside one trajectory** — across
> conversation turns (`multiturn_jailbreak`), generated tokens (`trajguard`),
> cooperating agents (`cross_trajectory`), or a sparse campaign in a trace
> repository (`meerkat`). This lesson answers a different, **scale**-driven
> question: given a large, *evolving* policy taxonomy — dozens today, thousands
> tomorrow — how do you moderate every piece of content against **all** of the
> policies cheaply, and add a brand-new policy **without retraining**? The 2026
> answer is a **bi-encoder**: embed the content in one tower, embed each policy
> **description** in a second tower, cache the policy tower once, and score
> compatibility with a cosine. Per-request cost stops depending on the number of
> labels, and a new policy is one embed call away.

This is a **detection** lesson (no LLM judge — a classifier reads cosines off a
frozen embedder, exactly like lesson 1). Three methods are compared on a hard,
multi-dataset, many-label safety corpus: **bi_encoder** (the hero — cached,
zero-shot), **uni_encoder** (re-encode text+label per pair; accurate-ish but the
cost grows with the label count), and **trained_head** (supervised; strong on
seen labels, but structurally cannot score a policy it never trained on). The
whole point is a single set of orderings: the bi-encoder holds a **flat** cost as
labels grow *and* scores **held-out** policies zero-shot, where the trained head
scores nothing at all.

---

## The key idea in code

Two towers, one shared space. The policy tower is embedded **once** and cached;
each incoming text is embedded **once**; the decision is a dot product. Adding a
policy is embedding a sentence:

```python
# ---- build the POLICY tower ONCE and cache it (cost paid a single time) ------
policy_bank = embedder.encode([p["description"] for p in policies], kind="policy")
policy_bank /= np.linalg.norm(policy_bank, axis=1, keepdims=True)   # [P, dim], L2-normed

# ---- moderate incoming content: embed ONCE, then a single matmul -------------
content_vec = embedder.encode([text], kind="content")              # [1, dim], L2-normed
scores = content_vec @ policy_bank.T                               # [1, P] cosines
#  -> that matmul is the ENTIRE per-request cost. Doubling the number of policies
#     adds columns to policy_bank; it does NOT add encoder forward passes.

# ---- add a NEW policy ZERO-SHOT: no retraining, one embed call ---------------
new_vec = embedder.encode([new_policy["description"]], kind="policy")
new_vec /= np.linalg.norm(new_vec)
score_new = content_vec @ new_vec.T          # the new label scores immediately
```

Contrast the **uni-encoder**, which fuses text and label and must re-encode the
joint string for *every* (text, label) pair — so its cost is `n_texts x n_labels`
encoder calls, and it collapses past a few dozen policies. The bi-encoder pays
the policy cost once and reuses it forever. That difference is the entire lesson.
Full file-by-file walkthrough below.

---

## Table of contents

1. [Why bi beats uni at scale, and beats the trained head on unseen policies](#1-why-bi-beats-uni-at-scale-and-beats-the-trained-head-on-unseen-policies)
2. [The three methods](#2-the-three-methods)
3. [The two towers, in one diagram](#3-the-two-towers-in-one-diagram)
4. [Files](#4-files)
5. [Code walkthrough, file by file](#5-code-walkthrough-file-by-file)
6. [The dataset](#6-the-dataset)
   — [6.1 The Aegis 2.0 crosswalk](#61-the-aegis-20-crosswalk--and-where-we-added-a-column-instead-of-forcing-one)
   · [6.2 Three transfer arms, one OOD](#62-three-transfer-arms-and-only-one-of-them-is-ood)
   · [6.3 The confound audit](#63-the-confound-audit--four-bars-now-from-the-shared-spine)
7. [Hard-negative augmentation — the 2026 data-synthesis recipe](#7-hard-negative-augmentation--the-2026-data-synthesis-recipe)
8. [The safety-detection series](#8-the-safety-detection-series)
9. [Running](#9-running)
10. [Results — measured vs. the claim](#10-results--measured-vs-the-claim)
11. [Honest caveats](#11-honest-caveats)
12. [Repository](#12-repository)

---

## 1. Why bi beats uni at scale, and beats the trained head on unseen policies

A safety guardrail is a **many-label** classifier: a piece of content can violate
*animal abuse* and *violence* and *privacy* at once, and the list of policies you
care about grows every quarter as new abuse patterns appear. Two design axes
matter, and the three methods trade them off differently.

**The scaling axis.** A **uni-encoder** (a cross-encoder) is the accurate default:
it feeds the model the text **and** a policy description together — `"moderate:
{text}\npolicy: {desc}"` — and lets attention fuse them before reading a
compatibility score. But that fusion means the representation of the text depends
on *which* policy you paired it with, so you must run the encoder once **per
(text, policy) pair**. With `P` policies that is `P` forward passes per text; at a
thousand policies it is a thousand times the cost, and it does not fit a real-time
guardrail. The **bi-encoder** breaks the dependency: the text is embedded on its
own, each policy description is embedded on its own, and compatibility is a cheap
cosine. The policy vectors do not depend on the text, so they are computed **once
and cached** — per-request cost is one text embed plus a matmul, **flat** in the
number of labels (The Million-Label NER, arXiv:2602.18487; GLiNER
Guard, arXiv:2605.05277).

**The unseen-policy axis.** A **trained head** — a supervised one-vs-rest
classifier on the content embedding — is the strongest option **on the policies it
was trained on**. But it learns one weight vector per *seen* label, so a policy
that did not exist at training time has **no column**: it can score nothing, and
adds require collecting labels and retraining. The bi-encoder needs no label at
all — a new policy is a **description**, embedded into the same space, and it
scores immediately (**zero-shot**). That is the property a growing taxonomy
demands (Opir's 996-category taxonomy, arXiv:2605.29659).

So the bi-encoder is the design that is **cheap as labels grow** *and* **open to
new labels** — at the cost of some accuracy versus a cross-encoder that gets to
fuse text and label. This lesson measures exactly that trade: seen-policy accuracy
(where the trained head and uni-encoder should lead), zero-shot held-out accuracy
(where only the bi/uni-encoder can play at all), and latency versus label count
(where the bi-encoder should stay flat while the uni-encoder rises linearly).

---

## 2. The three methods

Each method turns a text and a set of policy columns into per-policy scores in
`[0, 1]` via `.fit(Xc_train, Y_train, seen_cols, policies)` then `.scores(Xc,
policy_bank, cols)`. `Xc` is precomputed **content** embeddings; `cols` selects
which policy columns to score.

| method (`config.METHODS`) | caches the policy tower? | scores an UNSEEN policy? | cost in #labels | role |
|---|---|---|---|---|
| `bi_encoder` | **yes** | **yes** (zero-shot from the description) | **flat** (one matmul) | **the hero** |
| `uni_encoder` | no (re-encode per pair) | yes, but re-encode every pair | **linear** (n_texts x n_labels) | the accurate-but-does-not-scale foil |
| `trained_head` | n/a | **no** (no column for an unseen label) | flat, but bounded to seen labels | the supervised ceiling on seen labels |

`bi_encoder` maps a content-vs-policy cosine to `[0, 1]` and calibrates a
per-column threshold on train — no weights are learned, so it works on **any**
column including held-out ones. `uni_encoder` embeds the joint `"moderate:
{text}\npolicy: {desc}"` string and a small logistic head reads the fused vector;
it can still score held-out policies (rebuild the joint string, reuse the head)
but never caches. `trained_head` fits one-vs-rest logistic on the content
embedding — strong on seen columns, `np.nan` (abstain) on any held-out column.

The pre-registered claims are **orderings**, not absolute numbers: (i) uni-encoder
latency **grows** with #labels while bi-encoder stays flat; (ii) bi-encoder
zero-shot macro-AP on held-out policies is **well above chance**; (iii) a
contrastive adapter **lowers** the bi-encoder's false-positive rate on hard
negatives. Each has a falsifier in [Section 10](#10-results--measured-vs-the-claim).

The backbone is **EmbeddingGemma-300M** (`google/embeddinggemma-300m`), a Gemma-3
based, 768-dimensional sentence embedder with Matryoshka truncation — the closest
open realization of the small embedding tower the cited papers describe. It is
trained with **task prompts**, so the content tower encodes with one prompt
(`query`) and the policy tower with another (`document`) — asymmetric retrieval,
handled inside `encoders.py`.

---

## 3. The two towers, in one diagram

```
        CONTENT TOWER (per request)                 POLICY TOWER (cached ONCE)
        ---------------------------                 --------------------------
   "How do I build a pipe bomb?"          policy 0  "violence: content depicting or
             |                                       enabling physical harm..."
             |                            policy 1  "drug_weapon: instructions to make
   embed with prompt_name="query"                    weapons or illicit drugs..."
             |                                 ...
             v                            policy P  "self_harm: content that encourages
       content_vec [dim]                            or instructs self-injury..."
             |                                       |
             |                            embed each with prompt_name="document"
             |                                       |  (done a SINGLE time)
             |                                       v
             |                                policy_bank [P, dim]  <-- cached to disk
             |                                       |
             +-------------------> cosine <----------+
                                     |
                                     v
                        scores [P]  in [0, 1]   (higher = policy applies)
                        add a NEW policy: embed its description -> one more row.
```

The content tower runs once per incoming text. The policy tower is embedded once
for the whole taxonomy and reused for every request; adding a policy appends one
row. The uni-encoder, by contrast, would rebuild the *left* side for every policy
on the right — which is why its cost tracks the label count.

---

## 4. Files

| file | role |
|---|---|
| `config.py` | every knob: embedder id + task prompts, the pooled datasets (BeaverTails, Aegis 2.0, toxic-chat; wildguardmix gated/unused), `N_PER_CLASS`/`N_BENIGN`, held-out policy count, method list, the hard-negative + multi-prototype + scaling settings, and all paths |
| `data.py` | build the many-label taxonomy; pool BeaverTails + Aegis 2.0 + toxic-chat into one multi-label corpus; the Aegis crosswalk; the seen/held-out policy split; group-aware train/test split; the three transfer arms; achieved-vs-requested provenance + `pool_fingerprint`; the four-bar confound audit (delegated to `common/confound.py`) |
| `encoders.py` | `get_embedder` (EmbeddingGemma / MiniLM), the multi-prototype `build_policy_bank`, the three guards (`BiEncoderGuard`, `UniEncoderGuard`, `TrainedHeadGuard`), the metrics, and the latency-vs-labels scaling microbenchmark |
| `hardneg.py` | the 2026 hard-negative module: dense mining, the ECIsem diagnostic, CausalNeg counterfactuals, the ARHN false-negative filter, and a small InfoNCE contrastive adapter over frozen embeddings |
| `run_biencoder_guard.py` | orchestrator: load -> embed (cached) -> fit the 3 guards -> EXP-A..EXP-F -> `results.json` + 4 PNGs |
| `infer.py` | build a tiny policy tower, match a harmful text and a benign hard-negative by cosine, then add a NEW policy zero-shot and watch it score — the lesson in one script |

---

## 5. Code walkthrough, file by file

### `config.py` — every knob in one place

The embedder id (`EMBED_MODEL = google/embeddinggemma-300m`, a local fallback
`EMBED_LOCAL`, `EMBEDDER` to switch to ungated MiniLM for a dry run), the
asymmetric task prompts (`CONTENT_PROMPT="query"`, `POLICY_PROMPT="document"`) and
Matryoshka `EMB_DIM`, the three data sources (`BEAVERTAILS_DATASET`,
`TOXICCHAT_DATASET`, `WILDGUARD_DATASET`), the rubric sizes (`N_PER_CLASS=500`,
`N_BENIGN=500`), the held-out policy count (`N_HELDOUT_POLICIES`), the method
list, and the three sub-modules' settings — multi-prototype (`POLICY_PARAPHRASES`,
`MULTIPROTO_ABLATION`), hard-negative (`HARDNEG_PER_POLICY`, `ADAPTER_*`,
`CONTRASTIVE_TEMP`), and scaling (`LABEL_SCALES`, `SCALE_BATCH`). Everything is
overridable by env var so an eval shrinks into one foreground window (see
[Running](#9-running)).

### `data.py` — the many-label taxonomy, the pooled corpus, and the honesty checks

`build_taxonomy()` defines the label columns: the 14 BeaverTails harm categories
(animal_abuse, child_abuse, controversial_topics, discrimination_stereotype,
drug_weapon, financial_crime, hate_speech, misinformation, non_violent_unethical,
privacy_violation, self_harm, sexually_explicit, terrorism, violence) plus
adversarial/jailbreak and toxicity, plus the five columns Aegis 2.0 needs and no
existing policy covers (§6.1) — each a `Policy` with a one-sentence `description`
**and** several `paraphrases` (the synthetic-schema-expansion teaching point).

`load_corpus()` pools the public datasets into one multi-hot corpus over that
taxonomy: BeaverTails (14-way category dict → the core columns), **Aegis 2.0**
(`_load_aegis` + `_AEGIS_CROSSWALK_RAW`, a second independent taxonomy; an unsafe
row whose categories all fail to map is **dropped and counted**, never relabelled
benign), toxic-chat (in-the-wild toxicity/jailbreak positives, benign only as a
backfill), and wildguardmix (**gated — 0 rows on this host**).

`corpus_provenance()` records what the corpus **achieved** rather than what it was
asked for: per-column positives on the corpus *and* the test split, the realised
benign count and class balance, the measured `source_distribution`, a per-column
`shortfall` flag, and a `pool_fingerprint` (SHA-256 over the sorted normalised
texts). `format_shortfall()` prints a loud banner whenever requested ≠ achieved —
a warning, not a failure, because a pool-limited column is legitimate and failing
to *say so* is not.

`split_seen_heldout()` withholds `N_HELDOUT_POLICIES` columns as **zero-shot**
policies never seen in training; `group_train_test()` is a group-aware split (no
text leakage). `load_transfer_arms()` builds the three separately-named arms of
§6.2 — `load_heldout_split()` (the BeaverTails shard formerly mislabelled `load_ood`,
which survives as a back-compat alias), `load_cross_annotator()` (Aegis' test split),
`load_ood_benchmark()` (CSTM-Bench). `confound_report()` delegates to the shared
four-bar audit in `common/confound.py` (§6.3).

### `encoders.py` — the towers, the guards, the metrics, the scaling benchmark

`get_embedder()` loads EmbeddingGemma (or MiniLM) **once, lazily**, and exposes
`.encode(texts, kind)` with `kind in {"content","policy"}` — routing to the right
task prompt, truncating to `EMB_DIM` (Matryoshka), and L2-normalizing rows.
`build_policy_bank()` is the **multi-prototype** tower: for each policy it embeds
the description **plus** its paraphrases, averages, and re-normalizes into one
robust policy vector (`n_proto=1` is the single-description ablation baseline).
The three guards implement the shared `.fit/.scores` contract from
[Section 2](#2-the-three-methods). `scaling_latency()` is the **million-label**
microbenchmark: for each label count `K` in `LABEL_SCALES`, it times the
bi-encoder (embed the texts **once** + matmul against `K` cached vectors) against
the uni-encoder (embed `K` joint strings **per** text) — the real taxonomy is
padded with synthetic `"policy N"` descriptions up to the largest `K`. Metrics are
standard: per-column average precision, macro/micro AP and F1, and any-policy
harmful AUC.

### `hardneg.py` — the 2026 hard-negative data-synthesis module

Operates on **precomputed** content embeddings and the policy bank (it does not
load the embedder). `mine_dense_hard_negatives()` retrieves, for each policy, the
benign texts with the **highest** cosine to that policy vector — the look-alikes
a boundary must learn to reject. `eci_score()` is the training-free ECIsem
diagnostic (arXiv:2603.20990) of a negative set in the frozen
geometry. `causal_counterfactuals()` builds CausalNeg-style
(arXiv:2606.01304) controlled negatives by violating exactly one
policy requirement with a **template** (no free-form generation).
`arhn_false_negative_filter()` (arXiv:2604.11092) drops a
candidate negative that actually still violates the policy. `ContrastiveAdapter`
is a small InfoNCE projection over the frozen vectors with adaptive hardness
weighting. Full narrative in [Section 7](#7-hard-negative-augmentation--the-2026-data-synthesis-recipe).

### `run_biencoder_guard.py` — the orchestrator

`main()` loads the corpus, the seen/held-out split, the group-aware train/test
split, and the three transfer arms; embeds all texts once (cached to disk per split
× embedder) and builds the policy bank; fits the guards on the **seen** columns;
then runs EXP-A (seen-policy multilabel AP/F1), EXP-B (held-out **zero-shot** — the
headline; the trained head reports `N/A`), EXP-C (the 1-vs-`P` prototype ablation),
EXP-D (the latency-vs-labels scaling curve), EXP-E (the three transfer arms), and
EXP-F (the hard-negative pipeline: mine → ECIsem → CausalNeg → ARHN → adapter,
comparing FPR@recall0.90 of the frozen bi-encoder vs. the adapter). It writes
`results.json` **before** the summary print and renders four PNGs (PR by method,
zero-shot bars, latency vs. labels, hard-neg FPR). Every EXP and plot is wrapped so
a late failure still leaves the data on disk.

Two caches are **stamped, not counted.** `_encode_content_cached` and
`_build_bank_cached` used to accept a cached matrix on a row-count match alone, so
any change preserving the count — a new dataset mix, an edited policy description —
silently reused vectors belonging to different text. They now store a SHA-256
fingerprint of the ordered inputs and **reject** on mismatch, including when a cache
predates fingerprinting and can prove nothing about itself (`cache REJECTED …`).
This is CLAUDE.md §18.8's "the embedding cache returned stale labels under a key
that ignored them", closed.

### `infer.py` — policy matching + the zero-shot demo, in one script

Builds a tiny policy tower from a handful of taxonomy policies, embeds a clearly
**harmful** text and a benign-but-adjacent **hard-negative** text with the content
tower, and prints the top policies each matches by cosine (the harmful text should
score a harm policy high; the adjacent benign text should score lower). It then
**adds a new policy** (`malicious_code`) the tower never saw — by embedding its
description alone — and shows a malware request lights it up while the benign text
does not, with zero retraining. All model-touching code is under `main()` with
lazy imports.

---

## 6. The dataset

One pooled, multi-label safety corpus over a shared policy taxonomy, plus three
separately-named transfer arms and held-out zero-shot policies on top.

> ### What the previous version of this section claimed, and what actually happened
>
> This section used to describe a three-way pool in the indicative. The 2026-08
> audit re-ran the loader and measured the corpus that produced the numbers in §10:
>
> ```
> source dist: beavertails=4852, beavertails_benign=500, toxicchat=374
> [data] wildguardmix SKIPPED (gated/unavailable: ... error 403)
> corpus N=5726  policies=16  benign=500
> ```
>
> **BeaverTails was 93.5% of the corpus. toxic-chat was 6.5%. wildguardmix was 0%.**
> Two sentences here were false in that run and are now deleted:
>
> - *"toxic-chat supplies … the benign hard-negatives that look adversarial but are
>   safe."* **It supplied zero benign rows.** The toxic-chat benign draw is gated
>   behind `deficit = max(0, n_benign - bt_benign)`, and BeaverTails filled the
>   entire 500-row quota first, so `tc_benign` was **0 by construction** whenever
>   `N_BENIGN <= 500`. Every benign row in that corpus was a BeaverTails safe row —
>   the one property this lesson advertised for its negatives was not in the data.
> - *"wildguardmix supplies adversarial-but-benign prompts."* **wildguardmix has
>   never loaded on this host.** It is gated, the host has no HF token, and the
>   request returns HTTP 403. It has contributed **zero rows of any kind** to every
>   corpus this lesson has ever built. The loader is kept so a tokened host gets it;
>   no result here rests on it, and this is stated in the indicative, not as a
>   conditional.
>
> Neither failure was visible from `results.json`, which recorded no source
> distribution and no per-column counts. That is fixed: every run now writes
> `source_distribution`, `achieved` and a `pool_fingerprint` (§6.1).

| role | dataset (loader) | what it is | contributes |
|---|---|---|---|
| **core taxonomy** | **BeaverTails** `PKU-Alignment/BeaverTails` `30k_train` (`data.py`) | prompt+response with a 14-way harm-category dict, multi-label | the 14 core policy columns + the length-matched benign pool |
| **second taxonomy** | **Aegis 2.0** `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` (`data.py`) | 33,416 rows, 12 core + 9 fine-grained categories, prompt **and** response, separate human/LLM label columns; **ungated** | an independent annotation regime, the starved columns, thousands of extra safe rows, 5 new policy columns |
| **in-the-wild hard** | **toxic-chat** `lmsys/toxic-chat` `toxicchat0124` (`data.py`) | real user prompts with toxicity + jailbreak flags | hard positives (toxicity/jailbreak); benign only as a backfill, and only if the earlier sources leave a deficit |
| **adversarial hard** | ~~**wildguardmix** `allenai/wildguardmix`~~ (`data.py`) | adversarial prompt-harm labels | **NOTHING on this host.** Gated, HTTP 403, no token. 0 rows in every run to date |
| **held-out policies** | columns of the taxonomy (`data.py`) | `N_HELDOUT_POLICIES` categories withheld from training | the **zero-shot** test — detected from the description alone |
| **transfer arms** | three, named separately (§6.2) | `heldout_split` / `cross_annotator` / `ood_benchmark` | only one of the three is out-of-distribution, and it says so |

**Why pool more than one dataset.** No single public set gives both a rich
many-label taxonomy **and** hard, in-the-wild adversarial text. But the deeper
reason is the one the audit found: a corpus that is 93.5% one dataset inherits that
dataset's annotation idiosyncrasies with no way to see it from the outside. Aegis
2.0 is the fix — a second, independently-annotated safety taxonomy over the same
kind of prompt+response text, large enough to fill the columns BeaverTails' 30k
split starves and to supply a benign pool big enough for hard-negative mining to
mean something. The label space is the **union** of these taxonomies, and every
policy carries a written description so the tower can match — and so an unseen
policy can be added by description alone.

### 6.1 The Aegis 2.0 crosswalk — and where we added a column instead of forcing one

Aegis' taxonomy is not ours. `violated_categories` is a `", "`-joined string of
category names, and `data._AEGIS_CROSSWALK_RAW` maps each one onto a **list** of
our policy ids (a list, because this corpus is multi-label and one Aegis category
can legitimately span two of our policies). Matching is done after normalisation,
so casing, punctuation and `Sexual (minor)` vs `Sexual Minor` resolve together.

| Aegis category | our column(s) | note |
|---|---|---|
| Hate/Identity Hate | `hate_speech` + `discrimination_stereotype` | split: the category covers both |
| Sexual | `sexually_explicit` | direct |
| Sexual (minor) | `child_abuse` + `sexually_explicit` | split |
| Suicide and Self Harm | `self_harm` | direct |
| Violence | `violence` | direct |
| Threat | `violence` | our `violence` policy names threats explicitly |
| Guns/Illegal Weapons | `drug_weapon` | direct |
| Controlled/Regulated Substances | `drug_weapon` | direct |
| PII/Privacy | `privacy_violation` | direct |
| Harassment | `toxicity` | our `toxicity` names harassing language |
| Profanity | `toxicity` | direct |
| Immoral/Unethical | `non_violent_unethical` | direct |
| Manipulation | `non_violent_unethical` | our description names manipulation |
| Fraud/Deception | `financial_crime` + `non_violent_unethical` | split |
| Political/Misinformation/Conspiracy | `misinformation` + `controversial_topics` | split: the category is genuinely two of ours |
| **Criminal Planning/Confessions** | **`criminal_planning` (NEW)** | no clean home; forcing it into `terrorism` or `financial_crime` would have been wrong |
| **Illegal Activity** | **`criminal_planning` (NEW)** | merged with the above — Aegis distinguishes them, we do not, and that is a documented loss |
| **Unauthorized Advice** | **`unauthorized_advice` (NEW)** | no existing policy covers unqualified medical/legal/financial advice |
| **Malware** | **`malware` (NEW)** | no cyber policy existed |
| **Copyright/Trademark/Plagiarism** | **`intellectual_property` (NEW)** | no IP policy existed |
| **High Risk Gov. Decision Making** | **`high_risk_gov_decisions` (NEW)** | no existing policy is about consequential state decisions |
| Needs Caution / Safe / None | *(nothing)* | top-level markers, not harm categories |

**Five columns were added rather than forced**, taking the taxonomy from 16 to 21.
They are **appended**, so indices 0–15 never move and every prior per-column result
stays comparable. Set `BG_AEGIS_EXTRA=0` to keep exactly 16 columns; the five
categories above then map to nothing.

**A row that maps to nothing is dropped and counted, never relabelled benign.** If
an Aegis row is flagged unsafe but every one of its categories fails to resolve, it
is excluded and tallied under `datasets.aegis.stats.unmapped_categories` in
`results.json`, with the exact unrecognised strings. Quietly turning an unsafe row
into a benign one is precisely the silent, plausible-looking failure CLAUDE.md
§18.8 catalogues, and it would have poisoned the negative pool.

**EXP-H's taxonomy is affected and says so.** Opir's `16 top + 126 mid + 854 leaf =
996` is exact only at 16 top-level policies. `build_taxonomy` **raises** rather than
mis-shaping, so the runner re-derives mid/leaf at the new top count while holding
the **996 total** and the 126:854 ratio (21/125/850), records both shapes under
`results['opir_shape']`, and prints the deviation. The total is the number Opir's
scaling claim is about; the exact 126/854 split is not.

### 6.2 Three transfer arms, and only one of them is OOD

The single arm this lesson used to call "OOD" was `BeaverTails/30k_test`: the same
dataset, the same annotators, the same 14-way taxonomy and the same prompt+response
rendering as 93.5% of train. **Only the rows changed.** That is split transfer, and
calling it out-of-distribution overstated it. It is kept — a held-out split is worth
measuring — and renamed so the name cannot overstate it again.

| arm | source | what actually shifts |
|---|---|---|
| `heldout_split` | `BeaverTails/30k_test` | **rows only.** Same dataset, annotators, taxonomy, rendering |
| `cross_annotator` | `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` `test` | **annotators + taxonomy.** Same prompt+response rendering |
| `ood_benchmark` | `intrinsec-ai/cstm-bench` | **everything.** A released external benchmark with a different corpus, task shape and label source |

**CSTM-Bench is the benchmark CLAUDE.md §17 rule 8 names for this lesson family**,
it is ungated, and it was already cached on this host while going unused. Three
limits are stated up front because the mapping onto a single-message moderation task
is a *choice*: (1) one row per **scenario**, sessions concatenated and capped, so the
label is scenario-level and a benign session inside an attack scenario is folded into
a positive; (2) only the `jailbreak` column and the binary harmful/benign score are
meaningful — the other columns are all-zero by construction and must **not** be read
as "the guard missed them"; (3) n ≈ 108, a screening-tier read on a genuinely
external distribution, not an evaluation-tier number.

**Held-out policies — the zero-shot test.** `N_HELDOUT_POLICIES` columns are
withheld from **all** training and are chosen from categories that still have
enough positives that the zero-shot number is real. Only the bi-encoder and
uni-encoder can score them; the trained head has no weight for them and reports
`N/A`. This is the experiment the whole design exists for.

### 6.3 The confound audit — four bars now, from the shared spine

A guardrail can look good for a boring reason. `confound_report()` no longer lives
in this lesson: it delegates to `steering_tutorials/common/confound.py`, the single
implementation every detection lesson shares. The local version measured **one** bar
(character length) and returned the **raw** AUC without folding it about 0.5 — one
of four partial reimplementations across the course, and the fold is not cosmetic
(a sibling lesson's `0.110` raw was a `0.890` confound with the sign flipped).

| bar | question | why it is needed |
|---|---|---|
| `length` | can raw character count separate the classes? | the classic length artifact |
| `count` | can word count separate them? | a token-count tell can hide behind a clean char count |
| `content` | can a 5-fold TF-IDF unigram model separate them? | **the bar that matters.** A policy-matching guard that cannot beat unigrams is not matching policies |
| `shuffle` | with labels permuted, does the pipeline still score above chance? | a leakage **diagnostic**, never a bar to clear |

Every bar is **directionless** — `max(auc, 1-auc)` — because a feature that predicts
the benign class perfectly is exactly as damning as one that predicts the harmful
class perfectly: a classifier learns either sign for free. The **binding bar** is the
largest of them, and `results.json` records the audit twice: over the whole corpus
(comparable to prior runs) and over the **test split**, which is where the methods
are scored and therefore where the bar a method must clear has to come from.
`results['margins']` states each method's `binary_harm_auc` minus that bar, with a
`clears` flag; a negative margin is printed, not hidden.

Legacy keys (`length_auc`, `len_pos_mean`, `len_neg_mean`) are still written so an
older reader does not break — but `length_auc` is now the **folded** value, with the
unfolded one beside it at `length_auc_raw`.

---

## 7. Hard-negative augmentation — the 2026 data-synthesis recipe

The single highest-leverage lever for a dual-encoder is not the architecture — it
is the **quality of its hard negatives**: the benign content that sits closest to
a policy and must be pushed away. A random benign text teaches nothing (it is
already far); a *look-alike* teaches the boundary. EXP-F walks the full 2026
recipe end-to-end, each stage a cited idea:

1. **Dense mining (ANCE-style).** Use the content tower itself to retrieve, per
   policy, the **benign** texts with the highest cosine to that policy vector —
   the true look-alikes, not random negatives. `mine_dense_hard_negatives()`.

2. **ECIsem pre-filter — measure the set *before* you train**
   ([arXiv:2603.20990](https://arxiv.org/abs/2603.20990)). A
   training-free diagnostic of a mined negative set in EmbeddingGemma's own frozen
   geometry: target-consistency (does the policy still prefer the positives?),
   semantic locality (how hard/close are the negatives?), a lexical-residual
   penalty (discount negatives that are merely token overlap), and diversity. A
   higher `eci` means a more *informative* negative set — so you can rank mining
   strategies without a single gradient step. `eci_score()`.

3. **CausalNeg counterfactuals — controlled, not free-form**
   ([arXiv:2606.01304](https://arxiv.org/abs/2606.01304)). Take a
   violating text, decompose the policy into requirements, and violate **exactly
   one** via a template (swap a harmful entity for a benign one, insert a negation
   or constraint, soften the ask) so the text stays fluent and on-topic but no
   longer violates. Templated string ops avoid the **generative-discriminative
   gap** — the failure mode where an LLM-*generated* negative is off-distribution
   from what a discriminator will actually see. `causal_counterfactuals()`.

4. **ARHN false-negative filter — do not poison the negatives**
   ([arXiv:2604.11092](https://arxiv.org/abs/2604.11092)). A mined
   or synthesized "negative" that actually **does** still violate the policy is a
   near-miss jailbreak mislabeled as safe — poison for the boundary. A
   policy-support check keeps a candidate negative only if it does **not** support
   the policy (our default is a cheap lexical stand-in for the paper's LLM
   answerability check). `arhn_false_negative_filter()`.

5. **Adaptive-weighted contrastive adapter.** A small projection (`ContrastiveAdapter`)
   trained with InfoNCE over the **frozen** content and policy vectors, weighting
   the loss toward the hardest **validated** negatives. It never touches the
   backbone — it sharpens the shared space. The test: does it **lower**
   FPR@recall0.90 on held-out hard negatives versus the frozen cosine?

The measured ECIsem summary, the counterfactual and dropped-false-negative counts,
and the frozen-vs-adapter FPR are all reported in
[Section 10, EXP-F](#10-results--measured-vs-the-claim).

---

## 8. The safety-detection series

This lesson is the **production-guardrail** member of a course that reads the same
"classify a signal off a frozen model" idea at widening scope:

| lesson | the unit it reads | the question it answers |
|---|---|---|
| [`multiturn_jailbreak`](../multiturn_jailbreak/README.md) | a conversation **turn** | is this escalating chat an attack? |
| [`trajguard`](../trajguard/README.md) | a generated **token** | is this completion drifting to harm? |
| [`cross_trajectory`](../cross_trajectory/README.md) | an agent **trajectory** | is a goal split across cooperating agents? |
| [`meerkat`](../meerkat/README.md) | a **repository** of traces | is a sparse campaign hiding in the fleet? |
| `biencoder_guard` (this lesson) | content vs. a **policy taxonomy** | which of *many* policies does this violate — cheaply, and for policies added yesterday? |

The first four ask "is *this* an attack?" This one asks "*which* of a thousand
policies, at constant cost, including ones you just wrote?" — the scale problem a
deployed guardrail actually faces.

---

## 9. Running

From the **repo root** (`steeringresearch/`):

```bash
# CPU-only self-tests (NO model, NO big download):
python -m steering_tutorials.biencoder_guard.encoders   # synthetic-embedding guard test
python -m steering_tutorials.biencoder_guard.hardneg    # synthetic hard-negative test
python -m steering_tutorials.biencoder_guard.data       # small corpus smoke + confound

# The full load -> embed -> fit -> EXP-A..F run (needs the EmbeddingGemma backbone):
python -m steering_tutorials.biencoder_guard.run_biencoder_guard

# Watch policy matching + the zero-shot new-policy demo:
python -m steering_tutorials.biencoder_guard.infer
```

**Env caps** (shrink an eval into one foreground window — the host's RAM, not
VRAM, is the wall):

| var | meaning | default |
|---|---|---|
| `BG_N_PER_CLASS` | positives per harm category | 500 |
| `BG_N_BENIGN` | benign hard-negatives | **3000** (was 500 — see below) |
| `BG_EMBED` | `embeddinggemma` or `minilm` (fast, ungated dry run) | `embeddinggemma` |
| `BG_PARAPHRASES` | descriptions averaged per policy (multi-prototype) | 4 |
| `BG_N_HELDOUT` | policy columns withheld for zero-shot | 4 |
| `BG_HARDNEG` | mined hard negatives per policy | 20 |
| `BG_AEGIS_ON` | pool `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` | 1 |
| `BG_AEGIS_EXTRA` | add the 5 policy columns Aegis needs (§6.1); 0 keeps 16 | 1 |
| `BG_TRANSFER_ARMS` | which transfer arms to run (§6.2) | `heldout_split,cross_annotator,ood_benchmark` |
| `BG_OPIR_AUTOFIT` | hold Opir's 996 total when the taxonomy is not 16 wide | 1 |

**Why `BG_N_BENIGN` went 500 → 3000.** At 500 the corpus was **5,226 harmful vs 500
benign — 91.3% / 8.7%**, and the harmful side was uncapped while the benign side was
hard-capped (`_load_beavertails` keeps a multi-label row while *any* of its columns
is under quota, so common columns overshot to 1,870 while `n_benign` stopped dead at
500). Two things broke as a result. First, `binary_harm_auc` and the confound bars
were computed on an 11:1 set with no prevalence-aware framing. Second — the sharper
one — the ANCE-style hard-negative miner draws `HARDNEG_PER_POLICY=20` per seen
policy = **240 negatives from the ~350** in the train split, i.e. **~69% of the whole
pool**. Dense mining means "retrieve the look-alikes"; retrieving two-thirds of
everything is not mining, and that is why EXP-F's frozen baseline sits at
`FPR@recall0.90 = 1.000`, the literal maximum.

3000 is chosen so that the selection is a **small fraction** of the pool rather than
most of it: the train split then holds ~2,100 benign rows and 240 mined is ~11% of
them — a ~6× larger pool to be selective within — while the class balance moves to
roughly **64% / 36%**. BeaverTails `30k_train` and Aegis 2.0 both have thousands of
unused safe rows, so nothing about the old cap was pool-limited; it was arbitrary.
Raise it further with `BG_N_BENIGN`; the cost is linear in embedding time.

```bash
# a fast, ungated smoke on MiniLM with a small corpus:
BG_EMBED=minilm BG_N_PER_CLASS=80 BG_N_BENIGN=80 \
  python -m steering_tutorials.biencoder_guard.run_biencoder_guard
```

On Windows PowerShell set env vars first, e.g. `$env:BG_EMBED = "minilm"`.

**No judge.** This is a **detection** lesson: a classifier reads cosines off a
frozen embedder, exactly as in lesson 1. There is no generation and no LLM judge,
so the off-family-judge discipline of the steering lessons does not apply here
(`results.json` records `"judge": null`).

---

## 10. Results — measured vs. the claim

> ### PROVENANCE OF EVERY NUMBER IN §10 (read before the tables)
>
> **UPDATED 2026-08-21.** Every figure below is read from
> `artifacts/results_embeddinggemma.json`, written by the re-run on the **fixed
> bidirectional** EmbeddingGemma-300M. The previous version of this block described a
> 2026-07-31 run at 16 policy columns / `N_BENIGN=500` / N=5,726 and listed four axes on
> which the data layer had since moved. **All four of those moves are now executed**, so
> the block is rewritten to describe the corpus that was actually measured rather than
> the one that was pending:
>
> | axis | the superseded 2026-07-31 run | **the run below (2026-08-21)** |
> |---|---|---|
> | encoder attention | **causal** (the bug) | **bidirectional** (fixed) |
> | benign pool | 500 (mining drew 69% of it) | **3,000** |
> | sources | BeaverTails 93.5% / toxic-chat 6.5% / wildguard 0% | BeaverTails 4,852 + Aegis 2.0 **4,842** + toxic-chat 371 + 3,000 benign |
> | corpus size | N = 5,726 | **N = 13,065** (10,065 harmful / 3,000 benign; test split 3,910) |
> | policy columns | 16 | **21** (5 appended for unmappable Aegis categories, §6.1) |
> | transfer | one arm, called "OOD" | **three named arms, all executed** (§6.2) |
> | confound | one bar, unfolded | four bars, folded, corpus **and** test split (§6.3) |
>
> `allenai/wildguardmix` is still **GATED** on this host (HTTP 403 without a token) and
> contributes **0 rows**; the artifact records that under `datasets.wildguardmix`.
> **8 of the 21 policy columns fall short of the 500 requested** — `animal_abuse` 357,
> `child_abuse` 425, `terrorism` 293, `jailbreak` 106, `unauthorized_advice` 459,
> `malware` 149, `intellectual_property` 76, `high_risk_gov_decisions` 75 — which is a
> pool ceiling, recorded per column in `achieved.requested_vs_achieved`, not a sampling
> choice. No number below has been invented, adjusted, or projected forward.

> ### 10.0 The bi-encoder arm was measuring the wrong thing, and this is the fix
>
> **What was wrong.** `bi_encoder` scored with raw cosine between **frozen** content and
> policy embeddings. That is *not* what the papers this lesson cites actually do.
> GLiNER-bi-Encoder ([arXiv:2602.18487](https://arxiv.org/abs/2602.18487)) and GLiNER
> Guard ([arXiv:2605.05277](https://arxiv.org/abs/2605.05277)) **fine-tune** their label
> and context encoders with an InfoNCE-style contrastive loss and hard-negative mining.
> 2602.18487 says plainly that frozen encoders give only *"baseline performance"* and
> that task-adapted encoders *"significantly outperform static frozen representations."*
> So the weak numbers were real — but they were the numbers of a **degenerate ablation**
> being reported as the method.
>
> **The fix.** `ContrastiveBiEncoderGuard` learns two linear maps `W_content` and
> `W_policy` with multi-positive InfoNCE over the frozen backbone. Full backbone
> fine-tuning is out of budget on one laptop GPU, so it adapts the **space** rather than
> the backbone — but it keeps the property that makes a bi-encoder a guard: a policy is
> scored **from its text, never its index**, so unseen policies remain zero-shot.
> The untrained projection is initialised to reproduce frozen-space cosine, so any gain
> is attributable to training rather than a lucky random projection. Trained on **seen
> policies only**, so the held-out evaluation stays a genuine zero-shot test.
>
> > **Note on the initialisation — the default is a RANDOM ORTHONORMAL map, not `torch.eye`.**
> > The obvious way to "start at frozen cosine" is `torch.eye(d_in, dim)`, but that is a
> > **truncation**: it keeps dims 0..dim−1 and puts *exactly zero* weight on the remaining
> > 512 of 768 dimensions, which is only defensible for a Matryoshka-trained encoder whose
> > leading axes are informative by design. Measured on the real cached banks, against
> > frozen-space cosine: EmbeddingGemma mean |cos error| **0.1196** truncated vs **0.0322**
> > random orthonormal (**~3.7× worse**); MiniLM **0.0575** vs **0.0568** (random better on
> > both). Truncation wins slightly on *rank* correlation for the MRL model, as expected —
> > but this guard **calibrates per-column F1 thresholds on absolute scores**, so absolute
> > cosine error is what actually costs, and truncation distorts it ~4× more. A random
> > orthonormal map is a Johnson–Lindenstrauss near-isometry regardless of how the encoder
> > orders its axes, so it is the representation-agnostic default. `init="truncate"` is
> > retained as a **labelled ablation**. Every number in §10 below is from the
> > orthonormal-init run.
>
> | arm | EXP-A seen | EXP-B **unseen policy** | EXP-E `heldout_split` *(not OOD, see 6.2)* |
> |---|---|---|---|
> | `bi_encoder` (frozen cosine) | 0.227 | **0.385** | 0.246 |
> | `bi_encoder_trained` (InfoNCE) | **0.482** *(+113%)* | **0.178** *(−54%)* | **0.452** *(+83%)* |
> | `uni_encoder` | 0.156 | 0.129 | 0.204 |
> | `trained_head` | 0.553 | **N/A — no column exists** | 0.538 |
>
> *(macro-AP; **fixed bidirectional** EmbeddingGemma-300M, 500/class requested,
> 17 seen + 4 held-out policies. Held out: Financial Crime, Self Harm, Toxicity,
> Criminal Planning.)*
>
> ### The result splits in a way worth pausing on — and it survived the encoder fix
>
> Contrastive training **more than doubles** seen-policy AP (0.227 → 0.482) — reproducing
> the papers' central "trained ≫ frozen" claim. It nearly doubles AP under *row* shift
> (`heldout_split`, 0.246 → 0.452). But on *unseen policies* it does not merely degrade:
> it lands at **0.178 against frozen cosine's 0.385 — less than half**. Training does not
> fail to help there; it actively destroys most of the zero-shot ability the frozen
> backbone already had.
>
> **This split is the one result that reproduced across the causal-encoder bug.** The
> suspended run measured 0.240 / 0.575 / 0.182 / 0.382; the fixed run measures
> 0.227 / 0.482 / 0.178 / 0.385. The magnitudes moved — trained seen-AP fell ~0.09 — but
> the *shape* is unchanged: trained ≫ frozen on seen labels, frozen ≫ trained on unseen
> ones. A bug that plausibly explained the weak numbers turns out not to have produced
> the ordering, which is worth stating precisely because the suspension was written on
> the assumption that it might have.
>
> Those two shifts are different, and the split is the lesson:
>
> - **EXP-E `heldout_split` holds the policies fixed and changes only the ROWS.** Training
>   transfers across them. ✔ *(This was written as "changes the content"; the content
>   barely changes -- same dataset, annotators and rendering. The `cross_annotator` and
>   `ood_benchmark` arms test the stronger reading and have now **both run** — see EXP-E
>   below. Training still transfers under an annotator+taxonomy change, 0.175 → 0.351;
>   on the genuinely external `cstm-bench` every arm collapses to near chance.)*
> - **EXP-B holds the content distribution and changes the policies.** Training does not
>   transfer, and is *less than half* as good as frozen cosine. X
>
> **Why.** The projection is trained on **17 policies** (12 in the superseded run; the
> Aegis merge added five columns and the seen set grew with it). A 768→256 map fitted to
> seventeen policy vectors learns *those seventeen directions*, not a general notion of
> "text-matches-policy" — and note that going 12 → 17 did **not** rescue zero-shot
> transfer (0.182 → 0.178), which is the shape you would expect if the missing ingredient
> is label *scale* rather than a few more columns. The papers get zero-shot policy
> generalisation by training at
> **million-label scale** — that scale, not the architecture, is what buys the zero-shot
> property. On this host we can demonstrate the trained-vs-frozen gap honestly; we cannot
> manufacture the label diversity that makes it generalise.
>
> **So the corrected headline for this lesson is:** a bi-encoder's zero-shot ability is a
> property of **label-scale training**, not of the bi-encoder architecture. Reporting the
> frozen arm as "the bi-encoder" understated the method; reporting the trained arm as
> zero-shot-capable at 12 policies would overstate it. Both numbers are now shown side by
> side, which is the only honest presentation of the two.
>
> *(Kept deliberately: `bi_encoder` frozen cosine remains as the labelled ablation it
> actually is, so the comparison stays visible instead of being silently replaced.)*

> ### 10.1 The label-scale test (EXP-G) — and a metric that cannot answer the question
>
> The paper's scaling experiment runs to **~1000 labels** and makes **two** claims:
> latency stays flat (already tested here by `scaling_latency`), and **accuracy is
> maintained**. EXP-G tests the second: pad the 21 real policies with **879 synthetic
> distractor policies** and re-score at K = 21 / 64 / 256 / **900**, computing metrics
> over the 21 real columns only.
>
> **First, a warning that changes how you read the table.** `macro_AP` and `macro_F1` are
> **exactly** flat across K — spread `0.00e+00`, asserted in code. That is not a
> measurement, it is **arithmetic**: a bi-encoder's score for column *j* is
> `cos(content, bank[j])`, which does not depend on any other column. Adding 879
> competitors cannot move a per-column average by construction. **Anyone reporting that
> flatness as "accuracy is maintained at scale" would be reporting a tautology.**
>
> The question only has content under a **competitive** rule — where distractors can
> outrank the true policy:
>
> | arm | K=21 | K=64 | K=256 | **K=900** | degradation |
> |---|---|---|---|---|---|
> | `bi_encoder` mean rank of true policy | 5.20 | 8.20 | 22.07 | **66.62** | **12.8×** |
> | `bi_encoder_trained` mean rank | **3.93** | **6.73** | **19.69** | **61.56** | 15.7× |
> | *(chance)* | 11.0 | 32.5 | 128.5 | 450.5 | 41× |
> | `bi_encoder` **rank / chance** | 0.473 | 0.252 | 0.172 | **0.148** | improves |
> | `bi_encoder_trained` **rank / chance** | **0.357** | **0.207** | **0.153** | **0.137** | improves |
> | `bi_encoder` **distractor steal@1** | 0.000 | 0.086 | 0.215 | **0.266** | worsens |
> | `bi_encoder_trained` **steal@1** | 0.000 | **0.006** | **0.017** | **0.033** | worsens |
>
> > **REVERSAL — this table's conclusion is the OPPOSITE of the suspended run's, and the
> > encoder fix is what moved it.** The causal-encoder version of this section read *"the
> > trained arm starts (barely) better and ends much worse"* (162.49 vs 62.55 at K=900,
> > rank/chance 0.361 vs 0.139) and explained it as the projection over-fitting to its
> > seen policies. **On the fixed bidirectional encoder the trained arm is better at every
> > single K**, and its distractor steal rate at K=900 is **0.033 against frozen cosine's
> > 0.266 — 8× more robust**. The old explanation was reasoning from an artifact. This is
> > the one place in the lesson where the bug did not merely shrink an effect, it inverted
> > one, which is why the suspension had to be a suspension rather than a footnote.
>
> **What survives from the old reading:** both arms still degrade in *absolute* rank
> (12.8× and 15.7× over a 43× increase in bank size) while *improving* relative to
> chance. So the paper's claim (2) is reproduced only in the weak relative sense, and the
> competitive picture at K=900 is a mean true-policy rank in the sixties out of 900.
>
> **Two further honesty notes recorded in the artifact:**
> - `top-3 F1` *rises* with K, which is an artifact: distractors consume top-3 slots, so
>   real predictions per row fall 3.00 → 1.71 (frozen) / 3.00 → 2.46 (trained) and
>   precision improves for free. It is excluded from the plot and flagged as
>   `notes.top_t_f1_confound`.
> - Distractors are screened **lexically** (disjoint domain grid, harm-stem blocklist,
>   Jaccard ≤ 0.20 vs every real description *and* paraphrase; max observed 0.125), with
>   the embedding audit **report-only**. Filtering by cosine under the encoder being
>   tested would delete precisely the hard competitors and rig the result. Bank is
>   seed-12345 deterministic, fingerprint `471d78f2b0b1`, 879 of 1,080 candidates used,
>   prefix-nested so each smaller K is a strict prefix of the larger.
>
> **Verdict on the paper's claim (2):** *partially reproduced.* Both arms stay well above
> chance and both **improve** relative to chance as the bank grows, so accuracy is
> "maintained" in the relative sense; in absolute ranking both degrade, and the frozen arm
> now degrades *less* in rank while being far *worse* on top-1 steal. The paper trains at
> million-label scale — that scale is the thing we cannot replicate, and it remains the
> thing that would matter.

> ### 10.2 The Opir taxonomy test (EXP-H) — the same scale, but the competitors are *relatives*
>
> EXP-G's 884 distractors are **strangers by construction**: a disjoint content-operations
> domain, a harm-stem blocklist, and a hard gate that *drops* any candidate scoring above
> Jaccard 0.20 against a real policy. That makes EXP-G a test of whether the guard survives
> a bank full of obviously-off-topic policies.
>
> Opir ([arXiv:2605.29659](https://arxiv.org/abs/2605.29659)) trains against something much
> less forgiving: *"a three-level taxonomy containing 996 categories across 16 top-level
> labels, 126 mid-level labels, and 854 leaf labels."* In a taxonomy like that, a label's
> competitors are its own **siblings and descendants** — plausible near-misses, not strangers.
> `taxonomy.py` builds exactly that shape below our **21** real policies (the total is held
> at Opir's 996 and mid/leaf are re-derived at the new `n_top` by the paper's 126:854 ratio,
> giving **21 + 125 + 850 = 996**; `opir_shape.matches_paper` is therefore `false` and says
> so in the artifact). EXP-H re-runs EXP-G's protocol against it. Everything else is held
> fixed — same test rows, same two arms, same 21 scored columns, same prototype count, same
> fp32 backbone — so the manipulated variable is the **relatedness of the competitors**.
>
> **The construction is measured, not asserted.** Where `distractors.py` *drops* anything
> above Jaccard 0.20, `taxonomy.py` targets the opposite and reports what it got:
>
> | adjacency of the ~900 competitors | EXP-G (unrelated) | **EXP-H (related)** |
> |---|---|---|
> | mean lexical Jaccard to the true policy's own text | ≤ 0.20 *(hard gate; max observed 0.125)* | **0.381** *(min 0.158)* |
> | …vs. to the nearest **other** top-level policy | — | 0.073 → **5.19× closer to its own parent** |
> | share of competitors **above EXP-G's drop gate** | 0 *(by definition)* | **95.7 %** |
> | mean embedding cosine to a real policy | — | **0.721** *(to its own parent; 0.616 to the best other parent)* |
> | share lexically closest to its own parent | — | **100 %** (98.8 % in embedding space) |
>
> > **CORRECTED 2026-08-21 — the cosine-compression paragraph that used to sit here was an
> > artifact of the causal encoder and is withdrawn.** It read that EmbeddingGemma places
> > EXP-G's unrelated distractors at **0.765** mean cosine against **0.865** for EXP-H's
> > relatives, and concluded that "in this backbone's geometry cosines are compressed into a
> > narrow high band" so EXP-H was a large manipulation of semantics and a small one of
> > cosine. On the **fixed bidirectional** encoder the taxonomy bank sits at **0.721** to its
> > own parent versus **0.616** to the best other parent, `p95_max_cos_to_real` is 0.803, and
> > **nothing in the bank exceeds 0.90**. The band is not compressed; the causal encoder was
> > compressing it. Reading a *geometry* claim off a broken attention mask is exactly the
> > failure the suspension existed to catch, so the paragraph is deleted rather than patched.
>
> As in EXP-G, the embedding audit is **report-only**. Filtering by cosine under the encoder
> being tested would delete precisely the confusable relatives — which is the entire
> phenomenon EXP-H exists to measure. Bank fingerprint `3653ac41c687`, seed 20265, fully
> deterministic.
>
> #### Lead with the breakdown, not a steal rate
>
> A single "steal rate" would be a confident number that means nothing here, because the
> competitors are not interchangeable. When something outranks the true policy it is one of
> three quite different events:
>
> - a **descendant** — e.g. `Violence / threats of assault against a named individual`
>   beating `Violence`. The router picked a *narrower rule under the same policy*: an
>   over-specification, arguably not an error at all.
> - a **sibling** — a different top-level policy. A genuine mis-route.
> - an **other-branch cousin** — some *other* policy's mid or leaf. Also a genuine mis-route,
>   and the closest analogue of an EXP-G distractor stealing rank.
>
> Every share below is printed against the share a **uniformly random** competitor would
> produce, because the buckets differ ~15× in size (a policy has ~46 descendants but ~929
> cousins). Raw shares would be a picture of bucket sizes; only the enrichment has content.
>
> **What wins the argmax on a genuinely harmful row** (K = 996, n = **2,949** harmful rows;
> *share [chance] enrichment*):
>
> | arm | true top-level | **descendant** | sibling | other-branch |
> |---|---|---|---|---|
> | `bi_encoder` (frozen) | 0.114 [0.002] **61×** | **0.433** [0.086] **5.0×** | 0.081 [0.019] 4.2× | 0.372 [0.893] 0.42× |
> | `bi_encoder_trained` | **0.495** [0.002] **268×** | 0.149 [0.086] 1.72× | **0.215** [0.019] **11.2×** | 0.142 [0.893] 0.16× |
>
> **The two arms behave in completely different ways, and only the breakdown shows it.**
> The frozen arm hits the true top-level policy just 0.114 of the time, and its dominant
> outcome is **over-specification**: 43 % of its top-1 picks are the true policy's *own
> descendants*, enriched 5.0× over chance. The trained arm's dominant outcome is simply
> **correct** — 0.495 on the true top-level policy, 268× chance — and when it does miss, it
> misses by **mis-routing to a sibling top-level policy**, enriched **11.2×**. Collapsed
> into one "steal rate" these look like the same kind of error. They are not: the first is a
> router being too specific about the right policy, the second is a router choosing the
> wrong policy.
>
> *(One sub-claim did move with the encoder fix and is corrected rather than left standing:
> the trained arm used to be **depleted** on descendants at top-1, 0.67× chance. It is now
> **enriched** at 1.72×. The qualitative contrast — frozen over-specifies, trained
> mis-routes to siblings — survives; the "below chance on descendants" detail does not.)*
>
> Across the *whole* violation pool the same effect is present but diluted — descendants are
> enriched 1.46× (frozen) and **0.97×** (trained, i.e. at chance) — because ~93 % of the bank
> is cousins. Kinship concentrates **at the very top of the ranking**, which is exactly where
> a router reads.
>
> #### The ranking table
>
> | arm | mean rank of true policy | median | **excl. own descendants** | rank-1 rate | rank-1 excl. desc. |
> |---|---|---|---|---|---|
> | `bi_encoder` | 141.60 | 58 | **131.97** | 0.062 | **0.167** |
> | `bi_encoder_trained` | **88.64** | **8** | **84.65** | **0.269** | **0.319** |
> | *(chance)* | 498.5 | — | — | 0.001 | — |
>
> #### Head-to-head against EXP-G — and EXP-G's headline does not survive it
>
> Steal figures use **EXP-G's own predicate on both sides** (an *added* column wins the
> argmax). EXP-H's stricter "any column that is not a true top-level policy" rate is a
> different rule and is reported separately in the artifact, never against EXP-G.
>
> | arm | EXP-G rank (K=900, unrelated) | EXP-H rank (K=996, related) | EXP-H at **matched K=900** | ratio | EXP-G steal | EXP-H steal | EXP-H steal, descendants forgiven |
> |---|---|---|---|---|---|---|---|
> | `bi_encoder` | 66.62 | 141.60 | 131.72 | **1.98×** | 0.266 | 0.805 | **0.372** |
> | `bi_encoder_trained` | 61.56 | 88.64 | 85.88 | **1.40×** | 0.033 | 0.290 | **0.142** |
>
> > **REWRITTEN 2026-08-21 — this subsection previously turned on an ordering flip that no
> > longer exists.** It read *"the ordering flips, and that is the finding"*: on the causal
> > encoder the trained arm lost EXP-G (162.49 vs 62.55) and won EXP-H (169.08 vs 208.56),
> > and the whole argument was that EXP-G's headline had been a property of its distractors
> > being strangers. **On the fixed encoder the trained arm wins both**, so there is no flip
> > to explain. The claim is deleted, not softened.
>
> **What the fixed run shows instead is a clean relatedness cost that hits the two arms
> unequally.** Both degrade when strangers become relatives, but the frozen arm degrades
> **1.98×** at matched K against the trained arm's **1.40×**, and the gap in steal rate is
> the sharper statement: a related competitor takes the top-1 slot from frozen cosine
> **80.5 %** of the time versus **29.0 %** for the trained arm (**37.2 %** vs **14.2 %** if
> a descendant win is forgiven). Contrastive training buys robustness to *adjacency*
> specifically.
>
> That is mechanistically the expected direction, which is why it is worth more than the
> ordering the causal run produced: InfoNCE with hard negatives is *training to separate
> semantically adjacent things*. Relatives are exactly what it was fitted to reject. The
> old text reached a compatible conclusion by a route the data did not support; the
> conclusion is re-derived here from numbers that do.
>
> #### Two variables move here, not one — and the breakdown separates them
>
> Honesty requires saying that EXP-H changes the competitors along **two** axes at once:
> they are *kin of the truth*, and they are *harm-domain content at all* (EXP-G's were
> ordinary content-operations policies). The bucket breakdown is what lets the two be told
> apart:
>
> - **Domain effect.** The frozen arm's other-branch (cousin) top-1 rate alone is **0.372** —
>   already **1.4×** EXP-G's *entire* competitor steal rate of 0.266. Much of the added
>   difficulty is simply that the competitors are harm-flavoured, independent of kinship.
>   *(This multiple was 2.6× on the causal encoder, where EXP-G's steal rate was only 0.133;
>   the fixed encoder raises the EXP-G baseline, so the domain effect is a smaller share of
>   the total than the suspended run implied.)*
> - **Kinship effect.** On top of that, descendants are enriched **5.0×** and siblings
>   **4.2×** (frozen) / **11.2×** (trained) over chance at rank 1. That part *is* kinship.
>
> Reporting only the aggregate would have attributed all of it to relatedness. It is the
> same one-change-at-a-time discipline §10.3 applies to the MiniLM-vs-EmbeddingGemma
> comparison, applied to our own new experiment.
>
> #### Metrics that cannot answer the question (the EXP-G trap, twice over)
>
> As in EXP-G, `macro_AP` / `macro_F1` are **exactly** the EXP-A and EXP-G values
> (0.2569 frozen, 0.4243 trained; asserted in code, delta `0.0e+00`) — a bi-encoder's
> column-*j* score is `cos(content, bank[j])` and cannot notice the other 975 columns.
> EXP-H adds a **second** such pair: `mean_rank_true_top16` (5.198 / 3.931) and
> `top1_acc_top16` (0.576 / 0.660) reproduce EXP-G's K=21 row bit-for-bit, because
> restricting to the real columns deletes every competitor *either* experiment added. All
> four are reported **only as cross-experiment anchors** proving the two experiments score
> the same thing — the runner asserts them (`notes.cross_experiment_anchor_verified: true`)
> — and none is evidence about relatedness.
>
> `macro_f1_top_t` inherits EXP-G's confound and here it runs to its limit: the top-3 of 996
> columns contains a real policy only **0.35** times per row (frozen), so the metric collapses
> to 0.166 while telling you nothing about the encoder. Excluded from the plot.
>
> #### Verdict
>
> A **996-category taxonomy of relatives is a materially harder bank than a 900-label bank of
> strangers**, for both arms, and how much harder depends on whether you count a descendant
> win as an error. Three statements, each carrying its own qualifier:
>
> 1. **Frozen cosine degrades ~2× at matched K** when competitors become relatives. Its top-1
>    behaviour is dominated by its own descendants (0.433), so **whether this is a failure at
>    all depends on the deployment**: a router that acts on the *top-level* policy has been
>    mis-routed 0.805 of the time; one that acts on the matched *leaf* has been mis-routed
>    0.372 of the time. This experiment cannot settle which — that is a question about the
>    downstream system, not about the encoder — so both numbers are reported and neither is
>    promoted to *the* answer.
> 2. **Contrastive training degrades too, but less** (1.40× at matched K vs 1.98×), and it
>    leads the frozen arm at 88.64 vs 141.60 — a 1.60× lead. *(The causal-encoder run
>    reported the trained arm as wholly immune at 0.95×. That is withdrawn: it degrades,
>    just more slowly.)*
> 3. **Neither arm is usable as a 996-way router as-is.** Best true-top-level top-1 accuracy
>    is 0.495, and mean rank 88.64 against a chance line of 498.5 — well above chance, nowhere
>    near deployable.
>
> *Screening tier: one embedder, one seed, one corpus (n = 3,910 test rows, 5,435 positive
> pairs, 2,949 harmful rows). The competitor set is templated from handwritten narrowings,
> not Opir's real taxonomy, so this reproduces the label-space **shape** and the
> sibling/descendant **relation** — not that paper's categories.*




**MEASURED — the HEADLINE run: EmbeddingGemma-300M at 500/class, fixed bidirectional
encoder (2026-08-21).** Numbers below are from `artifacts/results_embeddinggemma.json`
for the real dual-encoder backbone (`google/embeddinggemma-300m`, 768-dim,
n_per_class = 500, n_benign = 3000). The earlier MiniLM@150 substitute run is retained in
§10.3 for comparison, and the superseded causal-encoder run in
`artifacts/results_CAUSAL_ENCODER_SUSPENDED.json`.

The falsifiers were pre-registered **before** the run, and **one of them trips** —
reported below without softening.

**The claim under test.** A bi-encoder that caches the policy tower moderates
against a large taxonomy at **flat** per-request cost and scores **unseen**
policies zero-shot from a description, where a uni-encoder's cost grows with the
label count and a trained head cannot score unseen policies at all (The
Million-Label NER, arXiv:2602.18487; GLiNER Guard,
arXiv:2605.05277).

**EXP-A — seen-policy multilabel** (test, seen columns; the trained head and
uni-encoder are expected to lead on accuracy here):

| method | macro-AP | micro-AP | macro-F1 | binary harm AUC |
|---|---|---|---|---|
| `bi_encoder` | 0.227 | 0.218 | 0.113 | 0.691 |
| `bi_encoder_trained` | 0.482 | 0.516 | 0.179 | 0.614 |
| `uni_encoder` | 0.156 | 0.116 | 0.182 | 0.586 |
| **`trained_head`** | **0.553** | **0.597** | **0.501** | **0.718** |

> **Read the AUC column against the binding confound bar, not against 0.5.** On this test
> split the **TF-IDF content bar is 0.804** and **no method clears it** — `trained_head`
> misses by **−0.086**, `bi_encoder` by **−0.113**. The length bar is a near-null 0.502, so
> length is not the problem; *lexical content* is. Pre-registered falsifier **(iv) FIRES**
> and no EXP-A number may be headlined. See §10.4.

**EXP-B — held-out ZERO-SHOT** (test, held-out policies; **the headline**):

| method | macro-AP | macro-F1 |
|---|---|---|
| **`bi_encoder`** | **0.385** | **0.165** |
| `bi_encoder_trained` | 0.178 | 0.195 |
| `uni_encoder` | 0.129 | 0.191 |
| `trained_head` | N/A (cannot score an unseen policy) | N/A |

*(Held-out policies: Financial Crime, Self Harm, Toxicity, Criminal Planning.)*

**EXP-C — multi-prototype ablation** (bi-encoder on held-out policies, 1 vs. `P`
prototypes):

| policy tower | macro-AP |
|---|---|
| single description (`n_proto=1`) | 0.354 |
| **multi-prototype (`n_proto=4`)** | **0.385**  (**+0.030**) |

**EXP-D — scaling: latency vs. #labels** (fixed text batch; seconds):

| #labels | `bi_encoder` (sec) | `uni_encoder` (sec) |
|---|---|---|
| 16 | 1.541 | 0.460 |
| 64 | 1.542 | 1.569 |
| 256 | 1.542 | 8.022 |
| **1024** | **1.543** *(flat)* | **36.11** |

> **RE-MEASURED 2026-08-21 (fixed encoder), and the RATIO is deliberately not quoted.**
> The bi-encoder is flat to **0.10 %** across a 64× increase in label count
> (1.5410 → 1.5426 s) while the uni-encoder rises **78×** (0.460 → 36.11 s). Both
> `gpu_witness` snapshots record `contended: false`, so by this lesson's own rule the
> numbers are quotable — **but the ratio is not**, and that is a rule this section now
> imposes on itself. `ratio_at_max_labels` for this one method has read **64× → 40.30× →
> 43.91× → 23.41×** across runs; CLAUDE.md §18.5 lists the first three as withdrawn, and a
> quantity whose run-to-run spread is 2.7× is not a finding. **The FLATNESS is the
> architectural claim, it is invariant across every run on record, and it is what this
> experiment establishes.** Any single ratio is a property of one machine state.
>
> **The crossover is also worth stating plainly, because it cuts against the marketing.**
> At 16 labels the uni-encoder is **3.3× FASTER** (0.460 s vs 1.541 s) — the bi-encoder
> pays a fixed content-tower cost that only amortises once the bank is large. The two cross
> between 16 and 64 labels. A bi-encoder is not cheaper at every scale; it is cheaper at
> *this problem's* scale, which is the honest version of the claim.
>
> The correction history below is kept in full, because the rule it establishes is what
> licensed quoting anything here at all.

> **CORRECTION (2026-07-31): this table previously read 0.422 → 0.424 s and 27.06 s,
> quoted as "64×". That ratio was wrong and was corrected to ~40×.** Those numbers were
> read directly from the `scaling` block of the then-committed (git HEAD)
> `artifacts/results.json`, which gave **40.3×**. A second, independent run — measured
> while another job shared the GPU — gave **39.2×**.
> The old `0.424 / 27.06` pair reproduces *no* artifact: no run on record has a
> `bi_encoder` of 0.424 *together with* a `uni_encoder` of 27.06.
>
> **The mechanism of the error is worth more than the correction.** These are wall-clock
> timings on a single shared GPU, and the retracted 64× came from dividing a *contended*
> uni-encoder measurement by an *uncontended* bi-encoder one — two different machine states
> inside a single ratio.
>
> **A second correction, to the first one.** This note originally went on to claim that the
> *ratio* is contention-robust, because contention appeared to scale both arms by nearly the
> same factor (1.76× and 1.71× across the only two runs then available). **A third
> measurement falsified that**, and it is left here rather than quietly edited out:
>
> | run | GPU state | `bi_encoder` @1024 | `uni_encoder` @1024 | ratio |
> |---|---|---|---|---|
> | prior committed run (2026-07-31) | quiet throughout | 0.392 s | 15.78 s | **40.30×** |
> | EXP-H run 1 | contended throughout | 0.690 s | 27.05 s | **39.23×** |
> | EXP-H run 2 | **changed mid-benchmark** | 0.732 s | 18.39 s | **25.13×** |
> | orthonormal-init regeneration (causal encoder) | quiet in both witnesses | 0.977 s | 42.92 s | **43.91×** |
> | **current artifact** (2026-08-21, fixed bidirectional) | quiet in both witnesses | **1.543 s** | **36.11 s** | **23.41×** |
>
> > **The fifth row is why this table now argues AGAINST quoting a ratio at all.** Five runs
> > of the *same method* have produced 64× / 40.30× / 39.23× / 25.13× / 43.91× / 23.41×. Two
> > of those (39.23×, 25.13×) were explained away as contention, and the rule that came out
> > of it — quote only a run that is `contended: false` in both witnesses — was supposed to
> > settle the matter. **It did not: 43.91× and 23.41× are BOTH quiet-in-both-witnesses runs
> > of the same code, and they differ by 1.9×.** The rule was necessary and insufficient.
> > What actually changed between them is the encoder's attention mode, which alters the
> > uni-encoder's per-pair work and the bi-encoder's one-time content pass by different
> > factors. So the operative rule is upgraded: **the ratio is not a quotable quantity in
> > this lesson under any conditions.** Quote the flat-vs-linear shape.
>
> **The `gpu_witness` block explains the outlier, and the explanation is damning for the
> benchmark rather than for the GPU.** Run 2 recorded `contended: true` *before* the
> measurement (1 co-tenant, 12.4 GB, 83 % util) and `contended: false` *after* it
> (0 co-tenants, 3.2 GB, 39 % util) — the co-tenant job **finished while EXP-D was
> running**. That matters because of how `scaling_latency` is built: `bi_encoder`'s time is
> dominated by a single text-embedding pass measured **once, before the K-loop**, while
> `uni_encoder` is timed **inside** the loop with K=1024 measured **last**. So run 2 divided
> a *contended* `bi` by a *largely uncontended* `uni`.
>
> That is **structurally the same error as the retracted 64×** — a ratio spanning two
> machine states — except that here it happened *inside a single run*. Being one run is no
> protection at all. The benchmark is intrinsically vulnerable to this, because its
> numerator and denominator are measured minutes apart; a fix would re-measure `bi`
> adjacent to each `uni` point, and is noted here as a known limitation rather than
> silently patched.
>
> **What this does and does not license — SUPERSEDED 2026-08-21.** This paragraph used to
> argue that the constant-state runs agreed to within ~12 % (40.30× / 39.23× / 43.91×) while
> only the state-changing run was an outlier at 25.13×, and concluded that the current
> artifact's 43.91× was "the single quotable figure." **The fixed-encoder run at 23.41× —
> quiet in both witnesses — falsifies that reading**, and the note above records the
> consequence: no ratio is quotable here. The paragraph is left in place, corrected, because
> it is the second time an n-small robustness inference about this number failed, and the
> pattern is more useful than the claim was.
>
> **The scaling claim itself is unaffected and still SURVIVES**, because it never depended
> on any of these ratios: `bi_encoder` is flat in **all five** runs (0.390→0.392,
> 0.689→0.690, 0.731→0.732, 0.974→0.977, and 1.5410→1.5426 on the current artifact — a
> 0.10 % spread across a 64× increase in label count) while `uni_encoder` grows ~linearly in
> all five. The **flat-vs-linear shape** is the robust finding; the absolute seconds are
> indicative only and the ratio is not quotable at all.
>
> **Scope of the contamination — measured, so it is not re-litigated later.** `scaling` is
> the *only* wall-clock block in this lesson, and therefore the only one contention can
> touch. Diffing a quiet run against a contended one, **every other block is bit-identical**:
> EXP-A, EXP-B, EXP-C, EXP-E, EXP-F, EXP-G and the confound audit all match exactly, and
> EXP-G's `steal@1` at K=900 agrees to `0.0e+00` (0.133290 in both). EXP-H is likewise pure
> cosine geometry, and its head-to-head reads EXP-G values computed **inside the same
> process on the same rows**, so it cannot mix machine states even in principle. **No
> accuracy number anywhere in §10 is contention-sensitive.**
>
> Every `scaling` block written from this point carries its own `gpu_witness` (co-tenant
> count, memory, utilisation, before *and* after the benchmark), `ratio_at_max_labels`, and
> a `contended` flag — *inside* the block, not annotated beside it, so a timing can never
> again be quoted without its conditions travelling with it.

**EXP-E — transfer** (train on the constructed set, evaluate with no further
fitting). **The arm below is `heldout_split`, NOT out-of-distribution** — it is
BeaverTails `30k_test`, the same dataset, annotators, taxonomy and rendering as
**60 % of train** (7,852 of 13,065 rows: 4,852 harmful + all 3,000 benign; it was
93.5 % before the Aegis merge), with only the rows changed (§6.2). It was labelled
"OOD content" here and that overstated it.

| method | binary harm AUC | macro-AP |
|---|---|---|
| `bi_encoder` | 0.655 | 0.246 |
| `bi_encoder_trained` | 0.683 | 0.452 |
| `uni_encoder` | 0.675 | 0.204 |
| **`trained_head`** | **0.743** | **0.538** |

**All three transfer arms have now RUN (2026-08-21).** The two that were `[PENDING RUN]`
here are executed and reported below — and they are the reason the `heldout_split` result
must not be read as generalisation.

| arm | source | n | what it shifts | status |
|---|---|---|---|---|
| `heldout_split` | `BeaverTails/30k_test` | 2,787 | **rows only** — same dataset, annotators, taxonomy, rendering | measured (table above) |
| `cross_annotator` | Aegis 2.0 `test` | 1,903 | **annotators + taxonomy**, same rendering | **measured** |
| `ood_benchmark` | `intrinsec-ai/cstm-bench` | 50 | **genuinely external** — different corpus, task shape, label source | **measured** |

**`cross_annotator`** (n = 1,903; 1,030 harmful):

| method | binary harm AUC | macro-AP |
|---|---|---|
| `bi_encoder` | 0.682 | 0.175 |
| `bi_encoder_trained` | **0.694** | 0.351 |
| `uni_encoder` | 0.439 *(below chance)* | 0.089 |
| `trained_head` | 0.657 | **0.445** |

**`ood_benchmark`** (n = **50**; 28 harmful; **one** scored column, `jailbreak`;
scenario-level labels):

| method | binary harm AUC | macro-AP |
|---|---|---|
| `bi_encoder` | **0.284** | 0.494 |
| `bi_encoder_trained` | 0.510 | 0.541 |
| `uni_encoder` | 0.396 | 0.527 |
| `trained_head` | 0.450 | 0.507 |

> **The gradient across the three arms is the actual result, and it is not flattering.**
> Change only the rows and the trained arm holds (0.452 macro-AP). Change the annotators
> and taxonomy and it drops to 0.351 but stays ahead of frozen cosine. Move to a genuinely
> external benchmark and **every arm collapses to chance**: three of four binary AUCs are
> *below* 0.5, and `bi_encoder`'s **0.284** is far enough below chance to be an inverted
> ranker rather than a noisy one. The macro-APs near 0.5 are not a rescue — with 28 of 50
> rows harmful the base rate *is* 0.56, so an AP of 0.494–0.541 is at or below chance.
>
> **Two limits on how hard that null may be pushed.** n = 50 is far under this course's
> ≥500/class floor and only **one** column is scorable, so the OOD arm is a directional
> signal, not a measurement — CLAUDE.md §17 rule 5 admits a released benchmark as OOD at
> whatever size it comes in, and this one is small. And the labels are *scenario*-level
> while the model scores *content*, a granularity mismatch recorded in the artifact under
> `label_granularity`. The honest reading is that this lesson has **no evidence of external
> generalisation**, not that it has evidence of failure.

**EXP-F — hard-negative augmentation** (frozen bi-encoder vs. the contrastive
adapter, on held-out hard negatives):

| quantity | value |
|---|---|
| ECIsem (`target_consistency` / `locality` / `lexical_residual` / `diversity` / **`eci`**) | −0.112 / 0.347 / 0.403 / 0.729 / **0.309** |
| FPR@recall0.90 — frozen bi-encoder | **1.000** |
| **FPR@recall0.90 — contrastive adapter** | **0.300** |
| **delta (frozen − adapter)** | **+0.700** (adapter is better) |
| hard negatives mined / counterfactuals / false-neg dropped | 340 / 20,164 / 76 |

> **The benign-pool fix was applied, and it did NOT move the floor.** `BG_N_BENIGN` was
> raised 500 → **3000** precisely so the miner would stop selecting most of its own pool:
> `n_mined = 340` = 17 seen policies × `HARDNEG_PER_POLICY=20`, now drawn from a train
> split of ~2,100 benign rows, so the selection is **~16 %** of the pool instead of ~69 %.
> That was the diagnosed cause. **The frozen baseline is still pinned at exactly 1.000.**
>
> So the previous note's explanation — that the miner was a near no-op and the frozen arm
> therefore had no boundary to sit on — is **falsified by its own fix**. A ten-fold larger,
> properly-mined negative pool leaves the frozen bi-encoder flagging *every* hard negative
> at the 90 %-recall threshold. That is a statement about frozen cosine on this corpus, not
> about the miner.
>
> **Falsifier (iii) was amended before this run to handle exactly this case**, and the
> amendment now binds: the frozen baseline must be **strictly below 1.000** or the result
> is recorded as **UNINFORMATIVE** rather than as a survival. It is 1.000. **The verdict is
> UNINFORMATIVE** — see §10.4. The +0.700 delta is real and is a win over a floor, and a
> win over a floor is not evidence for the method.

**Pre-registered falsifiers.** Three were registered before the original run and are
verdicted in §10.4. **Four experiments shipped a verdict with no falsifier written
before the run** — EXP-A, EXP-C, EXP-G and EXP-H; EXP-G and EXP-H were added after
the falsifier block and it was never extended. They were pre-registered here, dated
**2026-08-08**, **before** the Aegis/benign-raise re-run. **That run happened on
2026-08-21 and all seven are now verdicted in §10.4** — including one that fires.

- **(i) Scaling.** If uni-encoder latency does **not** grow with #labels while the
  bi-encoder stays flat (EXP-D), the scaling claim is **FALSE**.
- **(ii) Zero-shot.** If the bi-encoder's zero-shot macro-AP on held-out policies
  is **≤ 0.5** (chance-ish, EXP-B), the "add policies zero-shot from a
  description" claim is **FALSE**. *(This threshold was mis-specified — see §10.4.
  A corrected re-run must use `macro-AP >= 2x the held-out base rate`, which is the
  criterion pre-registered for the next run and does not retroactively rescue this
  one.)*
- **(iii) Hard-negative sharpening.** If the contrastive adapter does **not**
  lower FPR@recall0.90 versus the frozen bi-encoder on held-out hard negatives
  (EXP-F), then "hard-negative sharpening helps here" is **FALSE**. *(Amended
  2026-08-08, before the re-run: the frozen baseline must additionally be
  **strictly below 1.000**, or the comparison is against a floor and the result is
  recorded as UNINFORMATIVE rather than as a survival.)*

**Pre-registered 2026-08-08, before the re-run (previously missing):**

- **(iv) EXP-A, seen policies.** If no method's `binary_harm_auc` on the test split
  exceeds the **binding confound bar** (`results['confound']['binding_bar']` — the
  largest of the length / count / TF-IDF-content bars, directionless), then the
  claim that this corpus is separable by *policy matching* rather than by surface
  statistics is **FALSE**, and no EXP-A number may be headlined. Falsifier fires on
  `results['margins'][m]['clears'] == false` for every `m`.
- **(v) EXP-C, multi-prototype policy tower.** If averaging `POLICY_PARAPHRASES=4`
  paraphrases into each policy vector does **not** raise held-out zero-shot macro-AP
  above the single-description tower by **at least +0.01**, then "synthetic schema
  expansion helps here" is **FALSE**. *(The prior run measured +0.023, i.e. it would
  have survived — but that verdict was written after the fact and is not what a
  pre-registration is.)*
- **(vi) EXP-G, accuracy at label scale.** Per-column macro-AP is **arithmetically**
  invariant to K and cannot test anything (§10.1). The real claim is competitive:
  if the bi-encoder's `mean_rank_true / chance_rank` does **not** stay flat or
  improve as K grows 16 → 900, then "accuracy is maintained at scale" is **FALSE**
  for this backbone. Reporting the flat macro-AP as a scaling result is a
  **tautology** and is barred regardless of outcome.
- **(vii) EXP-H, related vs unrelated competitors.** If scoring against Opir-shaped
  **related** competitors (siblings and descendants) is **not** harder than against
  EXP-G's **unrelated** strangers at matched K — i.e. if `mean_rank_true` at K=900
  is not worse under EXP-H than under EXP-G for the `bi_encoder` arm — then the
  premise that taxonomic adjacency is what makes a large label bank hard is
  **FALSE**, and EXP-H measures nothing EXP-G did not.

No reclassification-after-the-fact, and no swapping to an easier condition to
rescue a failed ordering.


### 10.3 MiniLM@150 vs EmbeddingGemma@500 — and why this comparison is NOT clean

> **RENUMBERED 2026-07-31: this section was previously also numbered "10.1".** The file
> carried *two* sections with that number — the EXP-G label-scale test above and this one
> — which is not merely untidy: an instruction of the form "do not alter 10.1" becomes
> ambiguous, and that ambiguity actually occurred while the EXP-H section was being added.
> Renumbered to 10.3 so the next reader cannot inherit the trap. Cross-references
> elsewhere in this file were updated with it.

| quantity | MiniLM @150/class | **EmbeddingGemma @500/class (fixed, 2026-08-21)** |
|---|---|---|
| EXP-D scaling at 1024 labels | bi 0.044 s / uni 1.786 s | bi **1.543** s / uni **36.11** s |
| EXP-B zero-shot (bi, macro-AP) | 0.408 | **0.385** |
| EXP-A seen (bi, macro-AP) | 0.355 | **0.227** |
| EXP-F adapter FPR@recall0.90 | 0.850 → 0.613 | **1.000 → 0.300** |
| length confound | 0.517 | **0.505** |

*(The ratio column that used to sit in row 1 — 40.6× vs 43.9× — is removed. See EXP-D:
this ratio is not a quotable quantity, and comparing two of its values across backbones
was the specific error the note below withdraws.)*

**The bigger, purpose-built backbone scored LOWER on accuracy**, and the encoder fix did
not change that: 0.227 vs MiniLM's 0.355 on seen policies, 0.385 vs 0.408 zero-shot. That
is unexpected and it is reported rather than buried — but it must **not** be attributed to
the backbone, because this comparison changes **four things at once**:

1. the **encoder** (MiniLM → EmbeddingGemma-300M),
2. **n_per_class** (150 → 500) *and the benign pool* (500 → 3000),
3. **the corpus** (N=5,726 at 93.5 % BeaverTails → N=13,065 with Aegis 2.0 at 37 %), and
4. **which policies were held out** — the seen/held-out split is chosen from the
   populated columns, and the 21-column run selected *Financial Crime, Self Harm,
   Toxicity, Criminal Planning* instead of *Drug Weapon, Non Violent Unethical, Violence,
   Toxicity*. Different held-out policies have different base rates and different
   intrinsic difficulty, so the two zero-shot numbers are **not measuring the same task**.

That violates this course's own one-change-at-a-time rule, and the fixed-encoder re-run
made it *worse* rather than better by moving the corpus at the same time. **No backbone
conclusion may be drawn from it.** A clean test would pin the held-out column set, the
corpus and n_per_class and vary only the encoder — pre-registered here as the follow-up,
not claimed now.

What the change *does* establish cleanly is the **architectural** result: the bi-encoder's
cached policy tower keeps per-request cost **flat** in the label count while the
uni-encoder's grows linearly, on both backbones. That shape is backbone-independent by
construction, and it is the finding.

> **CORRECTION (2026-07-31) — this paragraph previously drew a stronger conclusion that
> the arithmetic does not support.** It read: *"the uni-encoder's cost grew from 43× to
> **64×** the bi-encoder at 1024 labels, precisely because a larger encoder makes every one
> of its `n_texts × n_labels` forward passes more expensive … **Scaling the backbone up
> makes the bi-encoder's advantage larger, not smaller.**"*
>
> Both ratios were wrong, and correcting them **removes the effect entirely**:
>
> | backbone | quoted ratio | ratio from the same row's own latencies | source |
> |---|---|---|---|
> | MiniLM @150 | 43× | **40.6×** (1.786 / 0.044) | arithmetic on this table's own figures |
> | EmbeddingGemma @500 (causal) | 64× | **43.9×** (42.924 / 0.9775) | `results_CAUSAL_ENCODER_SUSPENDED.json` |
> | EmbeddingGemma @500 (**fixed**) | — | **23.4×** (36.105 / 1.5426) | `results_embeddinggemma.json` (`scaling`) |
>
> **40.6× versus 43.9× is nothing like the claimed 43× → 64× growth**, so the claim that
> scaling the backbone up widens the bi-encoder's advantage is **not supported and is
> withdrawn**. In hindsight it should have been suspect on its face: both towers use the
> *same* backbone, so making that backbone more expensive multiplies the numerator and the
> denominator alike and largely cancels.
>
> **What must NOT be done is to revive the claim from the 3.3× residue, or to flip it into
> the opposite claim.** It is tempting to read 43.9× > 40.6× as a surviving trace of the
> growth effect, or to read them as close enough to prove the advantage is *invariant* to
> backbone size. Neither inference is available here, for two reasons. First, the MiniLM
> figure comes from a run whose GPU contention state was never recorded. Second, and
> decisively, the EXP-D correction note above shows this ratio moving **40.30× → 43.91×**
> across two quiet runs of the *same* backbone, and **40.30× → 25.13×** purely from a
> co-tenant job — swings of 3.6× and 15× respectively, both at least as large as the 3.3×
> gap being interpreted. A difference smaller than the measurement's own run-to-run spread
> is not evidence in either direction.
>
> So the honest state is: **the growth claim is withdrawn as unsupported, and no
> replacement claim about backbone size is made.** Settling it needs quiet-GPU measurements
> on both backbones with the `gpu_witness` block recording the state — pre-registered here
> as the follow-up, exactly as §10.3 does for the encoder comparison itself.

### 10.4 Verdicts against those falsifiers

**All seven are verdicted below against the 2026-08-21 fixed-encoder run. One FIRES.**

| falsifier | outcome | evidence |
|---|---|---|
| **(i) Scaling** | **SURVIVES — claim upheld** | bi-encoder is flat at **1.5410 → 1.5426 s** from 16 → 1024 labels (0.10 % spread); uni-encoder rises **0.460 → 36.11 s**. The predicted flat-vs-linear split is exactly what was measured. *(The ratio is deliberately not quoted — see EXP-D. It has read 64× / 40.30× / 39.23× / 25.13× / 43.91× / 23.41× across six runs and is not a quotable quantity. The verdict never depended on it.)* |
| **(ii) Zero-shot** | ⚠️ **TRIPS as literally written** | bi-encoder held-out macro-AP = **0.385 ≤ 0.5**. By the pre-registered rule, the claim is FALSE. The threshold is still mis-specified (see below) and still not renegotiated after the fact. |
| **(iii) Hard-negative sharpening** | **UNINFORMATIVE** *(the 2026-08-08 amendment fires)* | FPR@recall0.90 falls **1.000 → 0.300** (a 0.700 reduction) — but the amendment requires the frozen baseline to be **strictly below 1.000**, and it is exactly 1.000. Recorded as UNINFORMATIVE, not as a survival. The diagnosed cause (miner drawing ~69 % of the benign pool) **was fixed** — `BG_N_BENIGN` 500 → 3000, selection now ~16 % of pool — and the baseline did not move. See EXP-F. |
| **(iv) EXP-A, seen policies** | ❌ **FIRES — the falsifier is TRIGGERED** | **No method clears the binding confound bar.** Bar = TF-IDF **content** at **0.804** (test split, directionless). Margins: `trained_head` **−0.086** (0.718), `bi_encoder` **−0.113** (0.691), `bi_encoder_trained` **−0.190** (0.614), `uni_encoder` **−0.218** (0.586). `results['margins'][m]['clears'] == false` for **every** m, which is the literal firing condition. **No EXP-A number may be headlined.** |
| **(v) EXP-C, multi-prototype** | **SURVIVES** | 4 paraphrases raise held-out zero-shot macro-AP **0.354 → 0.385 = +0.030**, clearing the pre-registered **+0.01**. |
| **(vi) EXP-G, accuracy at label scale** | **SURVIVES** | `bi_encoder` mean_rank_true / chance **improves** monotonically as K grows 21 → 900: **0.473 → 0.252 → 0.172 → 0.148**. The claim "accuracy is maintained at scale" is not falsified under the competitive rule. *(The tautological flat macro-AP is excluded, as the pre-registration requires.)* |
| **(vii) EXP-H, related vs unrelated** | **SURVIVES** | At matched K=900, `bi_encoder` mean_rank_true is **131.72 under EXP-H vs 66.62 under EXP-G — 1.98× worse**. Taxonomic adjacency does make a large label bank harder, so EXP-H measures something EXP-G did not. |

> **(iv) is the most important line in this table and it should not be read past.** It says
> that a TF-IDF bag-of-words classifier separates this corpus's harmful rows from its benign
> rows **better than any of the four semantic methods do** (0.804 vs a best of 0.718). The
> length bar is a near-null **0.502** and the count bar is inert, so this is not a length
> artifact — it is lexical content doing the work. Everything in EXP-A is therefore a
> *within-family* comparison of methods that all sit below the surface-statistics bar, and
> the seen-label orderings in §10.0 must be read as such.
>
> **A prior reading of this line is WITHDRAWN.** An earlier note called the harm detector
> "close to useless" at ~0.59 against a 0.526 confound — a margin of ~0.06. On the fixed
> bidirectional encoder the binary harm AUCs are **0.718** (`trained_head`) and **0.691**
> (`bi_encoder`) against a **length** confound of **0.505**, a margin of roughly **0.21**.
> The detector is materially better than that reading said, and "close to useless" is not a
> fair description of it. **But the margin that matters is the one against the *binding*
> bar, not the length bar**, and against 0.804 every method is still short. Both halves go
> in the record: the old pessimism was wrong about the instrument, and the new numbers
> still do not clear the pre-registered bar.

**On (ii) — reporting the trip, and a specification error I will not hide behind.**
The falsifier is recorded as tripped because that is what the pre-registered rule
says, and rules are not renegotiated after seeing data. But the threshold itself was
**mis-specified**: `0.5` is a chance level for **AUC**, not for **average precision**.
AP's chance level is the **base rate**.

**UPDATED 2026-08-21 — the held-out policy set CHANGED with the Aegis merge, so the base
rates are recomputed rather than carried.** The 2026-08-08 correction computed chance-AP
for `animal_abuse / sexually_explicit / violence / toxicity`; at 21 columns the split
selects **`financial_crime` / `self_harm` / `toxicity` / `criminal_planning`**, and only
`toxicity` is common to both. Carrying the old 0.1344 forward would have been the exact
cross-run contamination this paragraph was written to stop. Measured on the current test
split (n = 3,910), from `achieved.per_column_positives_test`:

| held-out policy | positives / 3,910 | base rate |
|---|---|---|
| financial_crime | 295 | **0.0755** |
| self_harm | 211 | **0.0540** |
| toxicity | 402 | **0.1028** |
| criminal_planning | 469 | **0.1199** |
| **chance macro-AP** | — | **0.0880** |
| **the measured 0.385 is** | — | **4.37× chance** |

So the measured **0.385** is **4.37× chance** — a larger multiple than the superseded
run's 2.84×, though the two are not comparable head-to-head because the held-out set is
not the same set. It beats the uni-encoder's **0.129** and the trained projection's
**0.178** on the same policies, while the trained head cannot score them at all. Per-column
positive counts for **both** the corpus and the test split are written to the artifact
under `achieved.per_column_positives` / `achieved.per_column_positives_test`, so this
ratio is re-derivable from the file instead of carried between runs.

*(This paragraph previously also quoted **0.408** and **0.178** — the MiniLM@150
substitute run's figures — inside a section whose every other number is
EmbeddingGemma@500. That was corrected first; the base rates were the second half of
the same cross-run contamination.)*

So the *capability* is real; the *test of it* was written wrong. Both statements go
in the record: the pre-registered falsifier **tripped**, and the pre-registration
**contained an error** that a corrected re-run must fix by using a base-rate-relative
threshold (e.g. "macro-AP ≥ 2× chance"). Correcting a threshold **after** seeing the
data is exactly the HARKing this course forbids, so the corrected criterion applies
only to the *next* run — it does not retroactively rescue this one.

---

## 11. Honest caveats

- **No method clears the binding confound bar (2026-08-21).** The TF-IDF content bar on
  the test split is **0.804** and the best binary harm AUC is **0.718**. Pre-registered
  falsifier (iv) fires and no EXP-A number may be headlined. This belongs at the top of the
  caveat list because it outranks all of the ones below it — see §10.4.
- **There is no evidence of external generalisation.** The only genuinely out-of-corpus
  arm (`intrinsec-ai/cstm-bench`, n=50, one scored column) puts every method at or below
  chance. The `heldout_split` numbers change *rows only* and must not be read as OOD.
- **The encoder ran causal until 2026-08-17, and one conclusion inverted when it was
  fixed.** Every accuracy number here is from the re-run on the fixed bidirectional
  encoder. The seen-label ordering reproduced, but EXP-G's trained-vs-frozen conclusion
  **reversed** and EXP-H's embedding-geometry claim was withdrawn. The suspended artifact
  is kept at `artifacts/results_CAUSAL_ENCODER_SUSPENDED.json`; the diagnosis is in
  [`audits/AUDIT_2026-08-17_embeddinggemma_causal.md`](../../audits/AUDIT_2026-08-17_embeddinggemma_causal.md).
- **8 of 21 policy columns are pool-limited below the 500/class request** (`jailbreak`
  106, `high_risk_gov_decisions` 75, `intellectual_property` 76, `malware` 149,
  `terrorism` 293, `animal_abuse` 357, `child_abuse` 425, `unauthorized_advice` 459).
  Per-column metrics on those columns rest on small n. `allenai/wildguardmix` is gated on
  this host and contributes 0 rows.
- **Screening tier, not evaluation.** One embedder, one seed, a constructed corpus
  — a directional demo, not the n ≥ 7 seeds + rigor contract CLAUDE.md reserves
  the word "winner" for. Do not over-read the orderings.
- **A frozen general embedder is not a trained cross-attention guard.** The cited
  GLiNER family trains a bespoke encoder with cross-attention between text and
  label. We use EmbeddingGemma-300M **frozen** and off-the-shelf — a *general*
  sentence embedder, not a safety-tuned guardrail. Our numbers are a floor for
  what the pattern buys, not the papers' trained ceiling.
- **Our `uni_encoder` is a cross-encoder-*lite*.** It embeds the joint
  `text + policy` string with the same frozen bi-encoder and reads a small head —
  it does **not** reproduce GLiNER's trained token-level cross-attention. It stands
  in for the *scaling shape* of a cross-encoder (re-encode per label), not its
  exact accuracy.
- **Paraphrases are handwritten, not GPT-4.1-generated.** The multi-prototype
  "synthetic schema expansion" uses templated/handwritten restatements, so it is a
  weaker version of the paper practice; treat EXP-C as illustrative.
- **Inspired-by, not a paper reproduction.** The architecture (two frozen towers +
  cosine + a cached policy bank + zero-shot-by-description) operationalizes the
  *idea* the cited papers share; it is **not** a faithful reimplementation of any
  one paper's exact model. All cited arXiv ids are WebFetch-verified by the lead
  (see `AUDIT.md`).
- **The rubric floor is not met on every column, and the artifact now says which.**
  At the run in §10, six of sixteen columns held under 500 positives — `jailbreak`
  **109**, `child_abuse` 185, `self_harm` 205, `terrorism` 293, `animal_abuse` 357,
  `toxicity` 374 — while `results.json` recorded `"n_per_class": 500`, the
  *requested* value read straight off the config. `jailbreak` at 109 is the worst,
  and it is this lesson's own thesis column. Only `jailbreak`/`toxicity` were
  genuinely pool-limited (the toxic-chat ceiling); the four BeaverTails shortfalls
  were a **split** ceiling, not a dataset ceiling. Pooling Aegis 2.0 targets exactly
  those columns. Every run now writes `achieved.per_column_positives` (corpus and
  test split) and prints a loud shortfall banner, so the gap is auditable from the
  artifact instead of only from a re-run of the loader.
- **The lesson's headline arm is the *frozen* ablation on one axis and the *trained*
  arm on another.** §10.0 shows the split honestly, but it means no single row is
  "the bi-encoder"; read both.
- **EXP-H deviates from Opir's exact 16/126/854 split when Aegis' extra columns are
  on** (21/125/850, total held at 996). The paper's *total* is preserved; its exact
  mid/leaf split is not, and `results['opir_shape']` records both.

---

## 12. Repository

Source and full artifacts:
<https://github.com/dlmastery/steeringresearch/tree/master/steering_tutorials/biencoder_guard>

Cited: The Million-Label NER (arXiv:2602.18487), GLiNER Guard
(arXiv:2605.05277), Opir (arXiv:2605.29659),
GLiGuard (arXiv:2605.07982), EmbeddingGemma-300M; hard-negative
line ECIsem (arXiv:2603.20990), ARHN (arXiv:2604.11092),
CausalNeg (arXiv:2606.01304).

See also
[the course map](../README.md),
[the turn-level sibling — multiturn_jailbreak](../multiturn_jailbreak/README.md),
[the agent-level sibling — cross_trajectory](../cross_trajectory/README.md),
[the repository-scale sibling — meerkat](../meerkat/README.md), and
[lesson 1 — the single-prompt activation probe (READ)](../hello_world/README.md),
whose activation-reading idea this whole series generalizes.
