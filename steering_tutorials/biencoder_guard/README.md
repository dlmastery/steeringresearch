# Bi-Encoder Guardrail — cache the policy tower once, moderate a million labels for a cosine

> **Reference:** [The Million-Label NER: Breaking Scale Barriers with GLiNER bi-encoder (arXiv:2602.18487)](https://arxiv.org/abs/2602.18487) — Stepanov, Shtopko, Vodianytskyi, Lukashov (Feb 2026); [GLiNER Guard: Unified Encoder Family for Production LLM Safety and Privacy (arXiv:2605.05277)](https://arxiv.org/abs/2605.05277) — Minko, Sadiekh, Kokuykin (May 2026); [Opir: Efficient Multi-Task Safety Classification for Toxicity, Jailbreaks, Hate Speech, and Harmful Content (arXiv:2605.29659)](https://arxiv.org/abs/2605.29659) — Stepanov, Smechov (May 2026); [GLiGuard: Schema-Conditioned Classification for LLM Safeguard (arXiv:2605.07982)](https://arxiv.org/abs/2605.07982) — Zaratiana, Newhauser, Hurn-Maloney, Lewis (May 2026); the shared backbone [EmbeddingGemma-300M (model card)](https://huggingface.co/google/embeddinggemma-300m). Hard-negative data-synthesis line: [ECIsem: Semantic Residual Effective Contrastive Information for Evaluating Hard Negatives (arXiv:2603.20990)](https://arxiv.org/abs/2603.20990) — Sinha, Seetharaman, Bansal (Mar 2026); [ARHN: Answer-Centric Relabeling of Hard Negatives with Open-Source LLMs for Dense Retrieval (arXiv:2604.11092)](https://arxiv.org/abs/2604.11092) — Choi et al. (SIGIR 2026); [When Hard Negatives Hurt: Bridging the Generative-Discriminative Gap in Hard Negative Synthesis for Retrieval (arXiv:2606.01304)](https://arxiv.org/abs/2606.01304) — Zhang et al. (KDD 2026); method name **CausalNeg**.

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
> All figures below come from the **2026-07-31 run**: EmbeddingGemma-300M, **16**
> policy columns, `N_PER_CLASS=500`, `N_BENIGN=500`, corpus **N=5,726** at
> **93.5% BeaverTails**, and a single transfer arm that was mislabelled OOD. They
> are the numbers this lesson actually measured and they are not edited.
>
> Since then the data layer changed on four axes, so **these tables do not describe
> the corpus the code now builds** and must be re-measured:
>
> | axis | the run below | now |
> |---|---|---|
> | benign pool | 500 (mining drew 69% of it) | `BG_N_BENIGN=3000` |
> | sources | BeaverTails 93.5% / toxic-chat 6.5% / wildguard 0% | + Aegis 2.0 (33,416 rows, ungated) |
> | policy columns | 16 | 21 (5 appended for unmappable Aegis categories, §6.1) |
> | transfer | one arm, called "OOD" | three named arms, one genuinely OOD (§6.2) |
> | confound | one bar, unfolded | four bars, folded, corpus **and** test split (§6.3) |
>
> Anything marked **[PENDING RUN]** is wired and import-checked but has not been
> executed. No number below has been invented, adjusted, or projected forward.

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
> | `bi_encoder` (frozen cosine) | 0.240 | **0.382** | 0.184 |
> | `bi_encoder_trained` (InfoNCE) | **0.575** *(+140%)* | **0.182** *(−52%)* | **0.415** *(+126%)* |
> | `uni_encoder` | 0.169 | 0.115 | 0.146 |
> | `trained_head` | 0.658 | *abstains* | 0.496 |
>
> *(macro-AP; EmbeddingGemma-300M, 500/class, 12 seen + 4 held-out policies.)*
>
> ### The result splits in a way worth pausing on
>
> Contrastive training **more than doubles** seen-policy AP (0.240 → 0.575) — reproducing
> the papers' central "trained ≫ frozen" claim. It also **more than doubles** AP under
> *content* shift (BeaverTails OOD, 0.184 → 0.415). But on *unseen policies* it does not
> merely degrade: it lands at **0.182 against frozen cosine's 0.382 — less than half**.
> Training does not fail to help there; it actively destroys most of the zero-shot ability
> the frozen backbone already had.
>
> Those two shifts are different, and the split is the lesson:
>
> - **EXP-E `heldout_split` holds the policies fixed and changes only the ROWS.** Training
>   transfers across them. ✔ *(This was written as "changes the content"; the content
>   barely changes -- same dataset, annotators and rendering. The `cross_annotator` and
>   `ood_benchmark` arms are the ones that test the stronger reading, and they are
>   **[PENDING RUN]**.)*
> - **EXP-B holds the content distribution and changes the policies.** Training does not
>   transfer, and is *less than half* as good as frozen cosine. X
>
> **Why.** The projection is trained on **12 policies**. A 768→256 map fitted to twelve
> policy vectors learns *those twelve directions*, not a general notion of
> "text-matches-policy". The papers get zero-shot policy generalisation by training at
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
> maintained**. EXP-G tests the second: pad the 16 real policies with **884 synthetic
> distractor policies** and re-score at K = 16 / 64 / 256 / **900**, computing metrics
> over the 16 real columns only.
>
> **First, a warning that changes how you read the table.** `macro_AP` and `macro_F1` are
> **exactly** flat across K — spread `0.00e+00`, asserted in code. That is not a
> measurement, it is **arithmetic**: a bi-encoder's score for column *j* is
> `cos(content, bank[j])`, which does not depend on any other column. Adding 884
> competitors cannot move a per-column average by construction. **Anyone reporting that
> flatness as "accuracy is maintained at scale" would be reporting a tautology.**
>
> The question only has content under a **competitive** rule — where distractors can
> outrank the true policy:
>
> | arm | K=16 | K=64 | K=256 | **K=900** | degradation |
> |---|---|---|---|---|---|
> | `bi_encoder` mean rank of true policy | 4.46 | 7.59 | 19.80 | **62.55** | **14×** |
> | `bi_encoder_trained` mean rank | 4.36 | 12.98 | 47.19 | **162.49** | **37×** |
> | *(chance)* | 8.5 | 32.5 | 128.5 | 450.5 | 53× |
> | `bi_encoder` **rank / chance** | 0.525 | 0.234 | 0.154 | **0.139** | improves |
> | `bi_encoder_trained` **rank / chance** | 0.513 | 0.399 | 0.367 | **0.361** | improves |
>
> **The trained arm starts (barely) better and ends much worse.** At K=16 it ranks the
> true policy 4.36 vs 4.46 — a margin of a tenth of a rank. At K=900 it ranks it
> **162.5 vs 62.5**, **2.6× as deep**. Relative to chance the ordering flips too: frozen
> ends at 0.139, trained at 0.361.
>
> **This is the same weakness EXP-B found, seen from another angle.** The projection is
> fitted to 12 policies; 884 policies it has never seen are exactly what it has no
> machinery to reject. Frozen cosine has no such bias — it was never told which twelve
> mattered — so it degrades more gracefully among strangers. Trained wins where it was
> trained and loses where it was not, in both experiments.
>
> **Two further honesty notes recorded in the artifact:**
> - `top-3 F1` *rises* with K, which is an artifact: distractors consume top-3 slots, so
>   real predictions per row fall 3.00 → 2.20 and precision improves for free. It is
>   excluded from the plot and flagged as `notes.top_t_f1_confound`.
> - Distractors are screened **lexically** (disjoint domain grid, harm-stem blocklist,
>   Jaccard ≤ 0.20 vs every real description *and* paraphrase; max observed 0.125), with
>   the embedding audit **report-only**. Filtering by cosine under the encoder being
>   tested would delete precisely the hard competitors and rig the result. Bank is
>   seed-12345 deterministic, fingerprint `8c3fc96b2b05`, prefix-nested so each smaller
>   K is a strict prefix of the larger.
>
> **Verdict on the paper's claim (2):** *not reproduced at this label diversity.* Both
> arms stay well above chance, so accuracy is "maintained" only in the weak relative
> sense; in absolute ranking both degrade, and the trained arm degrades more. The paper
> trains at million-label scale — again, that scale is the thing we cannot replicate,
> and again it is the thing that would matter.

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
> `taxonomy.py` builds exactly that shape below our 16 real policies (7–8 mids per policy,
> 6–7 leaves per mid, **16 + 126 + 854 = 996**), and EXP-H re-runs EXP-G's protocol against
> it. Everything else is held fixed — same test rows, same two arms, same 16 scored columns,
> same prototype count, same fp32 backbone — so the manipulated variable is the
> **relatedness of the competitors**.
>
> **The construction is measured, not asserted.** Where `distractors.py` *drops* anything
> above Jaccard 0.20, `taxonomy.py` targets the opposite and reports what it got:
>
> | adjacency of the ~900 competitors | EXP-G (unrelated) | **EXP-H (related)** |
> |---|---|---|
> | mean lexical Jaccard to the true policy's own text | ≤ 0.20 *(hard gate; max observed 0.125)* | **0.381** *(min 0.150)* |
> | …vs. to the nearest **other** top-level policy | — | 0.067 → **5.7× closer to its own parent** |
> | share of competitors **above EXP-G's drop gate** | 0 *(by definition)* | **94.9 %** |
> | mean embedding cosine to a real policy | 0.765 *(max over the 16)* | **0.865** *(to its own parent; 0.805 to the best other parent)* |
> | share lexically closest to its own parent | — | **100 %** (99.9 % in embedding space) |
>
> **One row of that table deserves more attention than it first gets.** The *lexical*
> contrast between the two experiments is enormous — a 0.125 ceiling versus a 0.381 mean —
> but the *embedding* contrast is not: EmbeddingGemma already places EXP-G's deliberately
> unrelated content-operations distractors at **0.765** mean cosine to a real safety policy,
> against **0.865** for EXP-H's actual relatives. In this backbone's geometry, cosines are
> compressed into a narrow high band, and "obviously off-topic" is only ~0.1 further away
> than "sibling of the truth". So EXP-H is a large manipulation of *semantics* and a modest
> one of *cosine* — which is worth knowing before reading any of the degradations below as
> proportional to distance.
>
> As in EXP-G, the embedding audit is **report-only**. Filtering by cosine under the encoder
> being tested would delete precisely the confusable relatives — which is the entire
> phenomenon EXP-H exists to measure. Bank fingerprint `8611952dba23`, fully deterministic.
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
> produce, because the buckets differ ~15× in size (a policy has ~61 descendants but ~919
> cousins). Raw shares would be a picture of bucket sizes; only the enrichment has content.
>
> **What wins the argmax on a genuinely harmful row** (K = 996, n = 1,553 harmful rows;
> *share [chance] enrichment*):
>
> | arm | true top-level | **descendant** | sibling | other-branch |
> |---|---|---|---|---|
> | `bi_encoder` (frozen) | 0.074 [0.002] **43×** | **0.509** [0.104] **4.9×** | 0.069 [0.014] 4.8× | 0.348 [0.880] 0.40× |
> | `bi_encoder_trained` | **0.492** [0.002] **287×** | 0.070 [0.104] 0.67× | **0.308** [0.014] **21.4×** | 0.130 [0.880] 0.15× |
>
> **The two arms behave in completely different ways, and only the breakdown shows it.**
> The frozen arm hits the true top-level policy just 0.074 of the time, and its dominant
> outcome is **over-specification**: half its top-1 picks are the true policy's *own
> descendants*, enriched 4.9× over chance. The trained arm's dominant outcome is simply
> **correct** — 0.492 on the true top-level policy, 287× chance — and when it does miss, it
> misses by **mis-routing to a sibling top-level policy**, enriched **21.4×**, while being
> *depleted* on descendants (0.67×, i.e. below chance). Collapsed into one "steal rate"
> these look like the same kind of error. They are not: the first is a router being too
> specific about the right policy, the second is a router choosing the wrong policy.
>
> Across the *whole* violation pool the same effect is present but diluted — descendants are
> enriched 1.56× (frozen) and **0.74×** (trained, i.e. depleted) — because ~92 % of the bank
> is cousins.
> Kinship concentrates **at the very top of the ranking**, which is exactly where a router
> reads.
>
> #### The ranking table
>
> | arm | mean rank of true policy | median | **excl. own descendants** | rank-1 rate | rank-1 excl. desc. |
> |---|---|---|---|---|---|
> | `bi_encoder` | 208.56 | 98 | **188.81** | 0.043 | **0.195** |
> | `bi_encoder_trained` | **169.08** | **30** | **161.46** | **0.288** | **0.318** |
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
> | `bi_encoder` | 62.55 | 208.56 | 185.68 | **2.97×** | 0.133 | 0.857 | **0.348** |
> | `bi_encoder_trained` | 162.49 | 169.08 | 154.31 | **0.95×** | 0.026 | 0.200 | **0.130** |
>
> **The ordering flips, and that is the finding.** EXP-G concluded that the trained arm
> "starts better and ends worse" — 162.49 vs 62.55 at K=900 — and read that as the same
> weakness EXP-B found. Against *related* competitors the conclusion reverses: the frozen
> arm degrades **2.97×** while the trained arm does not degrade at all (**0.95×** — it
> actually ranks the truth *slightly better* among relatives than among strangers, 154.31 vs
> 162.49 at matched K=900), and the trained arm ends up ahead, **169.08 vs 208.56**. So
> EXP-G's headline was a property of its distractors being **strangers**, not a general
> property of contrastive training.
>
> **The margin is narrower than an earlier draft of this section reported.** With the
> orthonormal-init run the trained arm leads by **39.5 ranks (1.23×)** at K=996, and by
> **31.4 ranks (1.20×)** at matched K=900 — not the ~74-rank, 1.5× gap quoted when the
> projection was truncation-initialised. The *direction* of the flip is unchanged and so is
> the argument; only its size is smaller, and the smaller size is the one on record.
>
> That reversal is also mechanistically the expected one, which is why it is worth trusting
> more than the EXP-G ordering: InfoNCE with hard negatives is *training to separate
> semantically adjacent things*. Strangers are not what it was fitted to reject, and
> relatives are. The arm that looked worse on the easy test is the one that holds up on the
> hard one.
>
> #### Two variables move here, not one — and the breakdown separates them
>
> Honesty requires saying that EXP-H changes the competitors along **two** axes at once:
> they are *kin of the truth*, and they are *harm-domain content at all* (EXP-G's were
> ordinary content-operations policies). The bucket breakdown is what lets the two be told
> apart:
>
> - **Domain effect.** The frozen arm's other-branch (cousin) top-1 rate alone is **0.348** —
>   already **2.6×** EXP-G's *entire* competitor steal rate of 0.133. Much of the added
>   difficulty is simply that the competitors are harm-flavoured, independent of kinship.
> - **Kinship effect.** On top of that, descendants are enriched **4.9×** and siblings
>   **4.8×** (frozen) / **21.4×** (trained) over chance at rank 1. That part *is* kinship.
>
> Reporting only the aggregate would have attributed all of it to relatedness. It is the
> same one-change-at-a-time discipline §10.3 applies to the MiniLM-vs-EmbeddingGemma
> comparison, applied to our own new experiment.
>
> #### Metrics that cannot answer the question (the EXP-G trap, twice over)
>
> As in EXP-G, `macro_AP` / `macro_F1` are **exactly** the EXP-A and EXP-G values
> (0.2755 frozen, 0.4767 trained; asserted in code, delta `0.0e+00`) — a bi-encoder's
> column-*j* score is `cos(content, bank[j])` and cannot notice the other 980 columns.
> EXP-H adds a **second** such pair: `mean_rank_true_top16` (4.461 / 4.358) and
> `top1_acc_top16` reproduce EXP-G's K=16 row bit-for-bit, because restricting to the 16
> real columns deletes every competitor *either* experiment added. All four are reported
> **only as cross-experiment anchors** proving the two experiments score the same thing —
> the runner asserts them and records the verdict — and none is evidence about relatedness.
>
> `macro_f1_top_t` inherits EXP-G's confound and here it runs to its limit: the top-3 of 996
> columns contains a real policy only **0.27** times per row (frozen), so the metric collapses
> to 0.124 while telling you nothing about the encoder. Excluded from the plot.
>
> #### Verdict
>
> A **996-category taxonomy of relatives is a materially harder bank than a 900-label bank of
> strangers** — but only for the frozen arm, and only if you count a descendant win as an
> error. Three statements, each carrying its own qualifier:
>
> 1. **Frozen cosine degrades ~3× at matched K** when competitors become relatives. Its top-1
>    behaviour is dominated by its own descendants (0.509), so **whether this is a failure at
>    all depends on the deployment**: a router that acts on the *top-level* policy has been
>    mis-routed 0.857 of the time; one that acts on the matched *leaf* has been mis-routed
>    0.348 of the time. This experiment cannot settle which — that is a question about the
>    downstream system, not about the encoder — so both numbers are reported and neither is
>    promoted to *the* answer.
> 2. **Contrastive training is immune to the change** (0.95× at matched K — no degradation at
>    all) and overtakes the frozen arm, though by a narrower margin than an earlier draft
>    claimed: 169.08 vs 208.56, a 1.23× lead. EXP-G's contrary ordering does not generalise
>    past unrelated distractors.
> 3. **Neither arm is usable as a 996-way router as-is.** Best true-top-level top-1 accuracy
>    is 0.492, and mean rank 169.08 against a chance line of 498.5 — well above chance, nowhere
>    near deployable.
>
> *Screening tier: one embedder, one seed, one corpus (n = 1,710 test rows, 2,651 positive
> pairs). The competitor set is templated from handwritten narrowings, not Opir's real
> taxonomy, so this reproduces the label-space **shape** and the sibling/descendant
> **relation** — not that paper's categories.*




**MEASURED — the HEADLINE run: EmbeddingGemma-300M at 500/class.** Numbers below are
from `artifacts/results.json` for the real dual-encoder backbone
(`google/embeddinggemma-300m`, 768-dim, n_per_class = n_benign = 500). The earlier
MiniLM@150 substitute run is retained in §10.3 for comparison.

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
| `bi_encoder` | 0.240 | 0.243 | 0.152 | 0.589 |
| `uni_encoder` | 0.169 | 0.142 | 0.194 | 0.490 |
| **`trained_head`** | **0.658** | **0.667** | **0.595** | 0.588 |

**EXP-B — held-out ZERO-SHOT** (test, held-out policies; **the headline**):

| method | macro-AP | macro-F1 |
|---|---|---|
| **`bi_encoder`** | **0.382** | **0.223** |
| `uni_encoder` | 0.115 | 0.129 |
| `trained_head` | N/A (cannot score an unseen policy) | N/A |

**EXP-C — multi-prototype ablation** (bi-encoder on held-out policies, 1 vs. `P`
prototypes):

| policy tower | macro-AP |
|---|---|
| single description (`n_proto=1`) | 0.360 |
| **multi-prototype (`n_proto=4`)** | **0.382**  (**+0.023**) |

**EXP-D — scaling: latency vs. #labels** (fixed text batch; seconds):

| #labels | `bi_encoder` (sec) | `uni_encoder` (sec) |
|---|---|---|
| 16 | 0.974 | 0.983 |
| 64 | 0.985 | 3.743 |
| 256 | 0.984 | 11.366 |
| **1024** | **0.977** *(flat)* | **42.92** *(**43.9×** the bi-encoder)* |

> **UPDATE (regeneration run): the table above is re-measured, and the quotable ratio is now
> 43.91×.** The whole lesson was regenerated after the learned down-projection's init changed
> from truncation to random orthonormal (see §10.0), which re-ran `scaling` along with
> everything else. The figures above are read directly from the `scaling` block of the
> current `artifacts/results.json`, whose own `ratio_at_max_labels` is **43.91×**, measured
> with `contended: false` in **both** `gpu_witness` snapshots — the condition this lesson's
> own rule (below) requires before a ratio may be quoted. The absolute seconds are **2.5×**
> the previous run's on the bi-encoder and **2.7×** on the uni-encoder, which is what a
> whole-machine state difference looks like and is exactly why only same-state ratios are
> quotable. The correction history below
> is kept in full, because the rule it establishes is what licensed quoting this number.

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
> | **current artifact** (orthonormal-init regeneration) | quiet in both witnesses | **0.977 s** | **42.92 s** | **43.91×** |
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
> **What this does and does not license.** The *constant-state* runs agree to within ~12 %
> (40.30× quiet, 39.23× contended, 43.91× quiet on the current artifact) while the
> state-changing run is far off at 25.13×, which is consistent with the ratio being robust to
> a **steady** machine state and broken only by a **changing** one. That reading is
> mechanistically sensible — but it rests on **three** constant-state points, and an n=2
> robustness inference is precisely what failed here the first time. So it is recorded as the
> most plausible reading and **not asserted**. The operative rule stands regardless: quote
> only a run that is `contended: false` in **both** witnesses, which makes the current
> artifact's **43.91×** the single quotable figure.
>
> **The scaling claim itself is unaffected and still SURVIVES**, because it never depended
> on either number: `bi_encoder` is flat in **all four** runs (0.390→0.392, 0.689→0.690,
> 0.731→0.732, and 0.974→0.977 on the current artifact — a 1.1 % spread across a 64×
> increase in label count) while `uni_encoder` grows ~linearly in all four. The
> **flat-vs-linear shape** is the robust finding; the ratio is a quiet-GPU measurement and
> the absolute seconds are indicative only.
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
93.5% of train, with only the rows changed (§6.2). It was labelled "OOD content"
here and that overstated it.

| method | binary harm AUC | macro-AP |
|---|---|---|
| `bi_encoder` | 0.615 | 0.184 |
| `uni_encoder` | 0.571 | 0.146 |
| **`trained_head`** | **0.636** | **0.496** |

| arm | source | status |
|---|---|---|
| `heldout_split` | `BeaverTails/30k_test` | measured (the table above) |
| `cross_annotator` | Aegis 2.0 `test` | **[PENDING RUN]** — wired, not yet executed |
| `ood_benchmark` | `intrinsec-ai/cstm-bench` | **[PENDING RUN]** — wired, not yet executed |

**EXP-F — hard-negative augmentation** (frozen bi-encoder vs. the contrastive
adapter, on held-out hard negatives):

| quantity | value |
|---|---|
| ECIsem (`target_consistency` / `locality` / `lexical_residual` / `diversity` / **`eci`**) | −0.059 / 0.492 / 0.454 / 0.711 / **0.449** |
| FPR@recall0.90 — frozen bi-encoder | **1.000** |
| **FPR@recall0.90 — contrastive adapter** | **0.458** |
| **delta (frozen − adapter)** | **+0.542** (adapter is better) |
| hard negatives mined / counterfactuals / false-neg dropped | 240 / 8,985 / 22 |

> **The frozen baseline here is pinned at the worst value the metric can take, and
> falsifier (iii) therefore clears a floor.** `FPR@recall0.90 = 1.000` means that at
> the threshold needed for 90% recall the frozen bi-encoder flags **every single**
> hard negative. There is no value above 1.000, so *any* adapter that is not also
> degenerate produces a positive delta and the falsifier survives by construction.
> The +0.542 is real, and it is a win over a floor.
>
> **The cause is on the data side, not the method side.** `hardneg.n_mined = 240` =
> 12 seen policies × `HARDNEG_PER_POLICY=20`, drawn from a train split holding only
> ~350 benign rows (500 × 70%). "The 20 benign texts closest to each policy" was
> selecting **~69% of the entire pool** — ANCE-style dense mining is only meaningful
> when the pool is orders of magnitude larger than the selection, so the miner was a
> near no-op and the frozen arm had no boundary to sit on. `BG_N_BENIGN` has been
> raised 500 → **3000** for exactly this reason (§9); at that size 240 mined is ~11%
> of a ~2,100-row pool. **The numbers in this table are from the 500-benign run and
> must be re-measured.**

**Pre-registered falsifiers.** Three were registered before the original run and are
verdicted in §10.4. **Four experiments shipped a verdict with no falsifier written
before the run** — EXP-A, EXP-C, EXP-G and EXP-H; EXP-G and EXP-H were added after
the falsifier block and it was never extended. They are pre-registered here, dated
**2026-08-08**, **before** the Aegis/benign-raise re-run, and they are unverdicted
until that run exists.

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

| quantity | MiniLM @150/class | **EmbeddingGemma @500/class** |
|---|---|---|
| EXP-D scaling at 1024 labels | bi 0.044 s / uni 1.786 s = **40.6×** *(was quoted 43×)* | bi **0.977** s / uni **42.92** s = **43.9×** *(was quoted 64×, then 40.3× from an earlier run)* |
| EXP-B zero-shot (bi, macro-AP) | 0.408 | **0.382** |
| EXP-A seen (bi, macro-AP) | 0.355 | **0.240** |
| EXP-F adapter FPR@recall0.90 | 0.850 → 0.613 | **1.000 → 0.458** |
| length confound | 0.517 | 0.526 |

**The bigger, purpose-built backbone scored LOWER on accuracy.** That is unexpected and
it is reported rather than buried — but it must **not** be attributed to the backbone,
because this comparison changes **three things at once**:

1. the **encoder** (MiniLM → EmbeddingGemma-300M),
2. **n_per_class** (150 → 500), and
3. **which policies were held out** — the seen/held-out split is chosen from the
   populated columns, and at 500/class it selected *Animal Abuse, Sexually Explicit,
   Violence, Toxicity* instead of *Drug Weapon, Non Violent Unethical, Violence,
   Toxicity*. Different held-out policies have different base rates and different
   intrinsic difficulty, so the two zero-shot numbers are **not measuring the same task**.

That violates this course's own one-change-at-a-time rule. **No backbone conclusion may
be drawn from it.** A clean test would pin the held-out column set and n_per_class and
vary only the encoder — pre-registered here as the follow-up, not claimed now.

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
> | EmbeddingGemma @500 | 64× | **43.9×** (42.924 / 0.9775) | `artifacts/results.json` (`scaling`) |
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

| falsifier | outcome | evidence |
|---|---|---|
| **(i) Scaling** | **SURVIVES — claim upheld** | bi-encoder is flat at 0.974 → 0.977 s from 16 → 1024 labels; uni-encoder rises 0.983 → 42.92 s, **43.9×** the bi-encoder at 1024. The predicted flat-vs-linear split is exactly what was measured. *(Corrected 2026-07-31 from a mis-stated 64×, then re-measured at 43.91× by the orthonormal-init regeneration — see the EXP-D table. The verdict is unchanged across all of it; only the ratio moved.)* |
| **(ii) Zero-shot** | ⚠️ **TRIPS as literally written** | bi-encoder held-out macro-AP = **0.382 ≤ 0.5**. By the pre-registered rule, the claim is FALSE. |
| **(iii) Hard-negative sharpening** | **SURVIVES — against a floor** | FPR@recall0.90 falls **1.000 → 0.458** with the contrastive adapter (a 0.542 reduction). But **1.000 is the literal maximum the metric can take**: the frozen baseline flags *every* hard negative at the 90%-recall threshold, so any non-degenerate adapter produces a positive delta and this falsifier cannot fail. It is a true statement about a very low bar. Cause and fix in the EXP-F note above (the miner was drawing ~69% of the benign pool; `BG_N_BENIGN` 500 → 3000). **Re-measure before quoting.** |
| **(iv)–(vii)** | **[PENDING RUN]** | Pre-registered 2026-08-08, before the re-run. No verdict may be written until that run exists. |

**On (ii) — reporting the trip, and a specification error I will not hide behind.**
The falsifier is recorded as tripped because that is what the pre-registered rule
says, and rules are not renegotiated after seeing data. But the threshold itself was
**mis-specified**: `0.5` is a chance level for **AUC**, not for **average precision**.
AP's chance level is the **base rate**.

**CORRECTED 2026-08-08 — the base rates quoted here were wrong, and the correction
goes in this lesson's favour.** They read `0.118 / 0.180 / 0.342 / 0.102 → chance
macro-AP ≈ 0.185 → 2.1× chance`, carried over from the MiniLM@150 run with an honest
hedge that they were unverified. The hedge was right. Measured on the actual test
split at the shipped config (held-out = `animal_abuse`, `sexually_explicit`,
`violence`, `toxicity`):

| policy | previously quoted | **measured (test split)** |
|---|---|---|
| animal_abuse | 0.118 | **0.0591** |
| sexually_explicit | 0.180 | **0.0930** |
| violence | 0.342 | **0.3175** |
| toxicity | 0.102 | **0.0678** |
| **chance macro-AP** | 0.185 | **0.1344** |
| **the measured 0.382 is** | 2.1× chance | **2.84× chance** |

So the measured **0.382** is **2.84× chance**, and it beats the uni-encoder's
**0.115** (below chance) on the same policies while the trained head cannot score
them at all. The hedge is retired: per-column positive counts for **both** the
corpus and the test split are now written to `results.json` under
`achieved.per_column_positives` / `achieved.per_column_positives_test`, so this
ratio is re-derivable from the artifact instead of carried between runs.

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
