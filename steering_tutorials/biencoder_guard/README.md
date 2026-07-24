# Bi-Encoder Guardrail — cache the policy tower once, moderate a million labels for a cosine

> **Reference:** [The Million-Label NER: Bi-Encoder Named Entity Recognition at Scale (arXiv:2602.18487)](https://arxiv.org/abs/2602.18487) `[UNVERIFIED]`; [GLiNER Guard: A Unified Encoder Family for Production LLM Safety and Privacy (arXiv:2605.05277)](https://arxiv.org/abs/2605.05277) `[UNVERIFIED]`; [Opir: Efficient Multi-Task Safety Classification with a Three-Level Taxonomy (arXiv:2605.29659)](https://arxiv.org/abs/2605.29659) `[UNVERIFIED]`; [GLiGuard: Schema-Conditioned Classification for LLM Safeguard (arXiv:2605.07982)](https://arxiv.org/abs/2605.07982) `[UNVERIFIED]`; the shared backbone [EmbeddingGemma-300M (model card)](https://huggingface.co/google/embeddinggemma-300m). Hard-negative data-synthesis line: [ECIsem: Semantic Residual Effective Contrastive Information for Evaluating Hard Negatives (arXiv:2603.20990)](https://arxiv.org/abs/2603.20990) `[UNVERIFIED]`; [ARHN: Answer-Centric Relabeling of Hard Negatives with Open-Source LLMs for Dense Retrieval (arXiv:2604.11092)](https://arxiv.org/abs/2604.11092) `[UNVERIFIED]`; [CausalNeg: When Hard Negatives Hurt — Bridging the Generative-Discriminative Gap in Hard Negative Synthesis for Retrieval (arXiv:2606.01304)](https://arxiv.org/abs/2606.01304) `[UNVERIFIED]`.

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
number of labels (The Million-Label NER, arXiv:2602.18487 `[UNVERIFIED]`; GLiNER
Guard, arXiv:2605.05277 `[UNVERIFIED]`).

**The unseen-policy axis.** A **trained head** — a supervised one-vs-rest
classifier on the content embedding — is the strongest option **on the policies it
was trained on**. But it learns one weight vector per *seen* label, so a policy
that did not exist at training time has **no column**: it can score nothing, and
adds require collecting labels and retraining. The bi-encoder needs no label at
all — a new policy is a **description**, embedded into the same space, and it
scores immediately (**zero-shot**). That is the property a growing taxonomy
demands (Opir's 996-category taxonomy, arXiv:2605.29659 `[UNVERIFIED]`).

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
| `config.py` | every knob: embedder id + task prompts, the three datasets, `N_PER_CLASS`/`N_BENIGN`, held-out policy count, method list, the hard-negative + multi-prototype + scaling settings, and all paths |
| `data.py` | build the many-label taxonomy; pool BeaverTails + toxic-chat + wildguardmix into one multi-label corpus (>=500/class); the seen/held-out policy split; group-aware train/test split; OOD slice; the length-confound audit |
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
adversarial/jailbreak and toxicity — each a `Policy` with a one-sentence
`description` **and** several `paraphrases` (the synthetic-schema-expansion
teaching point). `load_corpus()` pools three public datasets into one multi-hot
corpus over that taxonomy: BeaverTails (14-way category dict → the core columns),
toxic-chat (real in-the-wild toxicity/jailbreak positives + adjacent benign hard
negatives), and wildguardmix (adversarial harmful prompts + benign-adversarial
hard negatives; skipped gracefully if it fails to load). `split_seen_heldout()`
withholds `N_HELDOUT_POLICIES` columns as **zero-shot** policies never seen in
training; `group_train_test()` is a group-aware split (no text leakage);
`load_ood()` renders a disjoint OOD shard to the same schema. `confound_report()`
is the honesty check: can raw character **length** alone separate harmful from
benign? An AUC near 0.5 means no trivial tell.

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
diagnostic (arXiv:2603.20990 `[UNVERIFIED]`) of a negative set in the frozen
geometry. `causal_counterfactuals()` builds CausalNeg-style
(arXiv:2606.01304 `[UNVERIFIED]`) controlled negatives by violating exactly one
policy requirement with a **template** (no free-form generation).
`arhn_false_negative_filter()` (arXiv:2604.11092 `[UNVERIFIED]`) drops a
candidate negative that actually still violates the policy. `ContrastiveAdapter`
is a small InfoNCE projection over the frozen vectors with adaptive hardness
weighting. Full narrative in [Section 7](#7-hard-negative-augmentation--the-2026-data-synthesis-recipe).

### `run_biencoder_guard.py` — the orchestrator

`main()` loads the corpus, the seen/held-out split, the group-aware train/test
split, and the OOD slice; embeds all texts once (cached to disk per split ×
embedder) and builds the policy bank; fits the three guards on the **seen**
columns; then runs EXP-A (seen-policy multilabel AP/F1), EXP-B (held-out
**zero-shot** — the headline; the trained head reports `N/A`), EXP-C (the
1-vs-`P` prototype ablation), EXP-D (the latency-vs-labels scaling curve), EXP-E
(OOD transfer), and EXP-F (the hard-negative pipeline: mine → ECIsem → CausalNeg
→ ARHN → adapter, comparing FPR@recall0.90 of the frozen bi-encoder vs. the
adapter). It writes `results.json` **before** the summary print and renders four
PNGs (PR by method, zero-shot bars, latency vs. labels, hard-neg FPR). Every EXP
and plot is wrapped so a late failure still leaves the data on disk.

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

One pooled, multi-label safety corpus over a shared policy taxonomy, built to the
project's hard-data rubric (>=500 positives **and** >=500 benign hard-negatives),
with a real OOD slice and held-out zero-shot policies on top.

| role | dataset (loader) | what it is | contributes |
|---|---|---|---|
| **core taxonomy** | **BeaverTails** `PKU-Alignment/BeaverTails` (`data.py`) | prompt+response with a 14-way harm-category dict, multi-label | the 14 core policy columns; thousands/category available |
| **in-the-wild hard** | **toxic-chat** `lmsys/toxic-chat` `toxicchat0124` (`data.py`) | real user prompts with toxicity + jailbreak flags | hard positives (toxicity/jailbreak) + topically-adjacent benign hard negatives |
| **adversarial hard** | **wildguardmix** `allenai/wildguardmix` (`data.py`) | adversarial prompt-harm labels | adversarial harmful positives + benign-adversarial hard negatives (skipped gracefully if gated) |
| **held-out policies** | columns of the taxonomy (`data.py`) | `N_HELDOUT_POLICIES` categories withheld from training | the **zero-shot** test — detected from the description alone |
| **OOD** | a disjoint shard (`data.py`) | same taxonomy, unseen rows | honest out-of-distribution transfer |

**Why pool three datasets.** No single public set gives both a rich many-label
taxonomy **and** hard, in-the-wild adversarial text. BeaverTails supplies the
fine-grained 14-way label structure the bi-encoder needs to *have many labels to
match against*; toxic-chat supplies genuine messy user prompts and the benign
hard-negatives that look adversarial but are safe; wildguardmix supplies
adversarial-but-benign prompts, the leakage-free style of hard negative. The label
space is the **union** of their taxonomies, and every policy carries a written
description so the tower can match — and so an unseen policy can be added by
description alone.

**Held-out policies — the zero-shot test.** `N_HELDOUT_POLICIES` columns are
withheld from **all** training and are chosen from categories that still have
enough positives that the zero-shot number is real. Only the bi-encoder and
uni-encoder can score them; the trained head has no weight for them and reports
`N/A`. This is the experiment the whole design exists for.

**The length-confound audit — and why it matters.** A guardrail can look good for
a boring reason: if harmful prompts are systematically longer than benign ones, a
length-only rule already separates the classes and no embedding understanding is
proven. `confound_report()` measures the AUC of raw character length against the
label. ≈0.5 → no shortcut, a high method AP reflects real policy-matching signal;
well above 0.5 → the headline must be discounted for a length artifact. The number
is recorded in `results.json` and quoted beside the method numbers.

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
   ([arXiv:2603.20990](https://arxiv.org/abs/2603.20990) `[UNVERIFIED]`). A
   training-free diagnostic of a mined negative set in EmbeddingGemma's own frozen
   geometry: target-consistency (does the policy still prefer the positives?),
   semantic locality (how hard/close are the negatives?), a lexical-residual
   penalty (discount negatives that are merely token overlap), and diversity. A
   higher `eci` means a more *informative* negative set — so you can rank mining
   strategies without a single gradient step. `eci_score()`.

3. **CausalNeg counterfactuals — controlled, not free-form**
   ([arXiv:2606.01304](https://arxiv.org/abs/2606.01304) `[UNVERIFIED]`). Take a
   violating text, decompose the policy into requirements, and violate **exactly
   one** via a template (swap a harmful entity for a benign one, insert a negation
   or constraint, soften the ask) so the text stays fluent and on-topic but no
   longer violates. Templated string ops avoid the **generative-discriminative
   gap** — the failure mode where an LLM-*generated* negative is off-distribution
   from what a discriminator will actually see. `causal_counterfactuals()`.

4. **ARHN false-negative filter — do not poison the negatives**
   ([arXiv:2604.11092](https://arxiv.org/abs/2604.11092) `[UNVERIFIED]`). A mined
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
| `BG_N_BENIGN` | benign hard-negatives | 500 |
| `BG_EMBED` | `embeddinggemma` or `minilm` (fast, ungated dry run) | `embeddinggemma` |
| `BG_PARAPHRASES` | descriptions averaged per policy (multi-prototype) | 4 |
| `BG_N_HELDOUT` | policy columns withheld for zero-shot | 4 |
| `BG_HARDNEG` | mined hard negatives per policy | 20 |

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

**[PENDING GPU RUN]** — the harness, the CPU self-tests, and this section's tables
are in place; the numbers are filled from `artifacts/results.json` once the
EmbeddingGemma run completes on the 4090. The falsifiers below are pre-registered
**before** the run.

**The claim under test.** A bi-encoder that caches the policy tower moderates
against a large taxonomy at **flat** per-request cost and scores **unseen**
policies zero-shot from a description, where a uni-encoder's cost grows with the
label count and a trained head cannot score unseen policies at all (The
Million-Label NER, arXiv:2602.18487 `[UNVERIFIED]`; GLiNER Guard,
arXiv:2605.05277 `[UNVERIFIED]`).

**EXP-A — seen-policy multilabel** (test, seen columns; the trained head and
uni-encoder are expected to lead on accuracy here):

| method | macro-AP | micro-AP | macro-F1 | binary harm AUC |
|---|---|---|---|---|
| `bi_encoder` | _pending_ | _pending_ | _pending_ | _pending_ |
| `uni_encoder` | _pending_ | _pending_ | _pending_ | _pending_ |
| `trained_head` | _pending_ | _pending_ | _pending_ | _pending_ |

**EXP-B — held-out ZERO-SHOT** (test, held-out policies; **the headline**):

| method | macro-AP | macro-F1 |
|---|---|---|
| `bi_encoder` | _pending_ | _pending_ |
| `uni_encoder` | _pending_ | _pending_ |
| `trained_head` | N/A (cannot score an unseen policy) | N/A |

**EXP-C — multi-prototype ablation** (bi-encoder on held-out policies, 1 vs. `P`
prototypes):

| policy tower | macro-AP |
|---|---|
| single description (`n_proto=1`) | _pending_ |
| multi-prototype (`n_proto=P`) | _pending_ |

**EXP-D — scaling: latency vs. #labels** (fixed text batch; seconds):

| #labels | `bi_encoder` (sec) | `uni_encoder` (sec) |
|---|---|---|
| 16 | _pending_ | _pending_ |
| 64 | _pending_ | _pending_ |
| 256 | _pending_ | _pending_ |
| 1024 | _pending_ | _pending_ |

**EXP-E — OOD transfer** (train on the constructed set, evaluate the disjoint OOD
shard with no further fitting):

| method | binary harm AUC | macro-AP |
|---|---|---|
| `bi_encoder` | _pending_ | _pending_ |
| `uni_encoder` | _pending_ | _pending_ |
| `trained_head` | _pending_ | _pending_ |

**EXP-F — hard-negative augmentation** (frozen bi-encoder vs. the contrastive
adapter, on held-out hard negatives):

| quantity | value |
|---|---|
| ECIsem summary (`target_consistency` / `locality` / `lexical_residual` / `diversity` / `eci`) | _pending_ |
| FPR@recall0.90 — frozen bi-encoder | _pending_ |
| FPR@recall0.90 — contrastive adapter | _pending_ |
| delta (frozen − adapter) | _pending_ |
| # CausalNeg counterfactuals built | _pending_ |
| # false-negatives dropped (ARHN) | _pending_ |

**Pre-registered falsifiers.**
- **(i) Scaling.** If uni-encoder latency does **not** grow with #labels while the
  bi-encoder stays flat (EXP-D), the scaling claim is **FALSE**.
- **(ii) Zero-shot.** If the bi-encoder's zero-shot macro-AP on held-out policies
  is **≤ 0.5** (chance-ish, EXP-B), the "add policies zero-shot from a
  description" claim is **FALSE**.
- **(iii) Hard-negative sharpening.** If the contrastive adapter does **not**
  lower FPR@recall0.90 versus the frozen bi-encoder on held-out hard negatives
  (EXP-F), then "hard-negative sharpening helps here" is **FALSE**.

No reclassification-after-the-fact, and no swapping to an easier condition to
rescue a failed ordering.

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
  one paper's exact model, and the cited ids are marked `[UNVERIFIED]` pending the
  lead's WebFetch check (see `AUDIT.md`).

---

## 12. Repository

Source and full artifacts:
<https://github.com/dlmastery/steeringresearch/tree/master/steering_tutorials/biencoder_guard>

Cited: The Million-Label NER (arXiv:2602.18487 `[UNVERIFIED]`), GLiNER Guard
(arXiv:2605.05277 `[UNVERIFIED]`), Opir (arXiv:2605.29659 `[UNVERIFIED]`),
GLiGuard (arXiv:2605.07982 `[UNVERIFIED]`), EmbeddingGemma-300M; hard-negative
line ECIsem (arXiv:2603.20990 `[UNVERIFIED]`), ARHN (arXiv:2604.11092
`[UNVERIFIED]`), CausalNeg (arXiv:2606.01304 `[UNVERIFIED]`).

See also
[the course map](../README.md),
[the turn-level sibling — multiturn_jailbreak](../multiturn_jailbreak/README.md),
[the agent-level sibling — cross_trajectory](../cross_trajectory/README.md),
[the repository-scale sibling — meerkat](../meerkat/README.md), and
[lesson 1 — the single-prompt activation probe (READ)](../hello_world/README.md),
whose activation-reading idea this whole series generalizes.
