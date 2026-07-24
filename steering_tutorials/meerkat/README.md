# Meerkat — the violation is 5% of the repository, and only clustering can see it

> **Reference:** [Detecting Safety Violations Across Many Agent Traces (arXiv:2604.11806)](https://arxiv.org/abs/2604.11806); the supervised sibling's benchmark [Cross-Session Threats in AI Agents: Benchmark, Evaluation, and Algorithms — CSTM-Bench (arXiv:2604.21131)](https://arxiv.org/abs/2604.21131); collusion-detection context [GroupGuard: A Framework for Modeling and Defending Collusive Attacks in Multi-Agent Systems (arXiv:2603.13940)](https://arxiv.org/abs/2603.13940); the trace embedder [BGE / C-Pack: Packed Resources for General Chinese/English Embeddings (arXiv:2309.07597)](https://arxiv.org/abs/2309.07597) `[UNVERIFIED]`; positives from [ActorAttack / Derail Yourself (arXiv:2410.10700)](https://arxiv.org/abs/2410.10700).

> The sibling lesson `cross_trajectory` **learned** a set-aggregator to catch a
> harmful goal split across K cooperating agents. This lesson takes the same
> "read the whole set, not one trace" instinct **unsupervised and at repository
> scale**. A safety violation in a deployed agent fleet is not a loud, single
> event — it is a **sparse (~5%), distributed campaign** hiding in a sea of benign
> traffic: a misuse campaign run across many sessions, a covert sabotage pattern,
> reward hacking repeated quietly. Score each trace **alone** (a per-trace monitor)
> and the diluted campaign never crosses any threshold. **Cluster** the embeddings
> of *many* traces and the campaign — whose members share an escalation shape —
> collapses into one tight cluster you can name and lift out whole. That is the
> Meerkat thesis: the signal lives in the *shape of the cloud*, not in any single
> point.

This is a **detection** lesson (no LLM judge — a localizer reads a signal off
frozen trace embeddings, exactly like lesson 1 and `cross_trajectory`). Positives
are real SafeMTData `Attack_600` decompositions rendered as traces; negatives are
benign UltraChat conversations rendered the same way. Three localizers are
compared across two base-rate regimes; the whole point is a single ordering: at
the **sparse** rate, **clustering the repository beats scoring each trace alone**.

---

## The key idea in code

Same trace embeddings, two ways to score a trace. The baseline asks "does **this
one trace** look like a violation?"; the Meerkat-style localizer asks "does this
trace sit in a **cluster that is enriched** for violations?":

```python
# PER-TRACE MONITOR (cluster.PerTraceMonitor) -- lesson-1's probe, one trace at a time:
p = logreg.fit(seed_emb, seed_labels).predict_proba(all_emb)[:, 1]  # P(violation | THIS trace)
#  -> a 5% campaign is DILUTED: no single diluted trace looks alarming on its own,
#     so the top-scored list fills with benign false positives. It MISSES.

# CLUSTER-ENRICHMENT (cluster.KMeansEnrich -- Meerkat-style):
labels = kmeans(all_emb, k=choose_k(all_emb))          # cluster the WHOLE repository
frac   = seed_violation_fraction_per_cluster(labels)   # label CLUSTERS from a few seeds
score  = frac[labels]                                  # a trace inherits its cluster's enrichment
#  -> the campaign's traces share an escalation shape, so they land in ONE cluster;
#     one seed label per cluster lifts the ENTIRE campaign at once. It SURFACES.
```

The per-trace monitor sees one dot; the clusterer sees the shape of the cloud. The
campaign is invisible to the first and obvious to the second — that difference is
the entire lesson. Full file-by-file walkthrough below.

---

## Table of contents

1. [Why per-trace fails and clustering catches it](#1-why-per-trace-fails-and-clustering-catches-it)
2. [The three localizers](#2-the-three-localizers)
3. [Pipeline](#3-pipeline)
4. [Files](#4-files)
5. [Code walkthrough, file by file](#5-code-walkthrough-file-by-file)
6. [The dataset](#6-the-dataset)
7. [The cross-trajectory series](#7-the-cross-trajectory-series)
8. [Running](#8-running)
9. [Results — measured vs. the claim](#9-results--measured-vs-the-claim)
10. [Honest caveats](#10-honest-caveats)
11. [Repository](#11-repository)

---

## 1. Why per-trace fails and clustering catches it

Picture a repository of a few hundred agent traces from a deployed assistant. The
vast majority are ordinary help requests. Buried among them, **5%** are a
coordinated misuse campaign — an adversary running the *same* ActorAttack-style
escalation across many sessions, each session an innocuous-looking chain of
sub-queries that only together reconstruct a harmful goal.

A **per-trace monitor** is lesson 1's probe applied one trace at a time: embed a
trace, ask a classifier `P(violation)`. It is **local by construction** — it can
only fire on a trace that looks harmful *in isolation*. But a well-built campaign
dilutes itself: each individual session is a plausible research or history chain,
so no single trace scores high enough to clear a threshold that would not also
drown the analyst in benign false positives. At a 5% base rate, ranking by
per-trace score puts a scatter of benign look-alikes at the top and the campaign
stays submerged. **The monitor is not broken — the signal is simply not in any one
trace.**

The signal is in the **collection**. The campaign's traces are not independent:
they were generated by the same recipe, so their embeddings cluster tightly
together, far from the diffuse benign cloud. **k-means over the whole repository**
recovers that structure without being told what to look for (this is the
unsupervised move — no seed scenario, no fixed workflow, per Meerkat,
arXiv:2604.11806). Once the campaign occupies one cluster, an analyst who labels a
**handful** of traces (the semi-supervised seed) discovers that cluster is, say,
90% violations — and **cluster-enrichment scoring** then lifts *every* member of
that cluster at once, campaign and all. What was 5% noise per-trace becomes one
90%-pure cluster you can name.

The benign twin has no such structure: ordinary traffic is topically scattered, so
benign traces spread across many low-enrichment clusters and never concentrate.
**The discriminator is cluster density, and only an aggregate view can see it.**

---

## 2. The three localizers

Each **trace** (one agent's rendered event sequence) embeds to one vector. Every
localizer exposes the same tiny API — `fit(emb, seed_idx, seed_labels)` then
`score(emb) -> [n]` in `[0, 1]` — so the run harness treats them interchangeably:

| method (`config.METHODS`) | uses cross-trace structure? | how it scores a trace | role |
|---|---|---|---|
| `per_trace` | **no** | logistic regression on the trace's own embedding (fit on the seed) | the diluted-campaign-defeated **baseline** |
| `kmeans_enrich` | **yes (clustering)** | k-means the whole repo → each trace inherits its cluster's seed-estimated violation fraction (Laplace-smoothed) | the **Meerkat-style headline** |
| `knn_purity` | yes (local density) | attack fraction among the trace's `KNN_K` nearest seed-labelled traces | a clustering-free density proxy |

All three see the **same** labelled seed — the only thing that differs is how they
generalize from it. `per_trace` generalizes by a decision surface on individual
embeddings; `kmeans_enrich` and `knn_purity` generalize by **who a trace sits
near**. The pre-registered claim is an **ordering**: at the sparse ~5% rate,
`kmeans_enrich` (and `knn_purity`) beat `per_trace` on **Average Precision**. If
they do not, "clustering surfaces the distributed campaign" is false (see the
falsifier in [Section 9](#9-results--measured-vs-the-claim)).

The embedder is the paper's own: **`BAAI/bge-base-en-v1.5`** (`bge`), a general
text embedder loaded via `transformers` + mean pooling (a faster `minilm`
substitute is available for quick runs). Unlike the other lessons, the signal here
is read off a **general sentence embedder**, not Gemma's residual stream — the
clustering, not the representation, is the object of study.

---

## 3. Pipeline

```
  SafeMTData Attack_600                 HuggingFaceH4/ultrachat_200k
  ActorAttack decompositions            benign multi-turn conversations
   ~5 sub-queries JOINED into            user turns JOINED into
   ONE trace string (label 1)            ONE trace string (label 0)
        |                                       |
        \___________ data.load_trace_pool()  ___/     (>=500 / class POOL)
                                |
                                v
             data.sample_repository(size, base_rate=0.05)
                    a SPARSE repository: ~5% attack, ~95% benign
                                |
             cluster.embed_traces(...)   bge-base-en-v1.5 mean-pool, L2-norm
                                |
                                v
                    emb : [n, dim]   (one vector per trace)
                                |
        +-----------------------+------------------------+
        |     label a small SEED (SEED_FRAC of traces)   |
        v                                                v
  per_trace          kmeans_enrich  (choose_k)        knn_purity
  (logreg/trace)     (cluster -> enrichment)          (k-NN density)   [cluster.py]
        |                    |                              |
        v                    v                              v
     score all traces -> Average Precision (sparse) + ROC-AUC (balanced)
                                |
              +-----------------+------------------+
              |  OOD: CSTM-Bench (real cross-      |
              |  session scenarios, arXiv:2604.21131)|
              v                                    v
         results.json  +  PCA-scatter / AP-vs-baserate / silhouette-vs-k PNGs
```

The per-trace-vs-cluster view — the per-trace monitor's top-scored traces (mostly
benign) beside the k-means cluster the campaign concentrates in — is what
`infer.py` prints.

---

## 4. Files

| file | role |
|---|---|
| `config.py` | every knob: embedder, data sources, pool size, base rate / repository sampling, k-grid, seed fraction, methods, paths |
| `data.py` | build the ≥500/class **trace pool** (Attack_600 decompositions + UltraChat benign); `sample_repository` at a target base rate; load OOD CSTM-Bench; the trace-length confound audit |
| `cluster.py` | the bge/minilm embedder, `choose_k` (silhouette), k-means, and the three localizers (`PerTraceMonitor`, `KMeansEnrich`, `KnnPurity`) + AP / ROC-AUC / cluster-purity metrics |
| `run_meerkat.py` | orchestrator: pool → embed → for each base rate build `N_REPOS` repositories → fit/score the 3 methods → cluster quality → OOD → `results.json` + plots |
| `infer.py` | build ONE small sparse repository; print the per-trace monitor's top traces (misses it) vs. the k-means cluster the campaign concentrates in (surfaces it) |

---

## 5. Code walkthrough, file by file

### `config.py` — every knob in one place

The embedder (`EMBED_MODEL = BAAI/bge-base-en-v1.5`, `EMBEDDER`, the `minilm`
fallback), the data sources (`SafeMTData/SafeMTData` `Attack_600`; `ultrachat_200k`
benign; the OOD `intrinsec-ai/cstm-bench`), the pool size (`N_ATTACK`/`N_BENIGN`,
≥500/class per the rubric), the **repository sampling** (`BASE_RATE = 0.05`,
`REPO_SIZE`, `N_REPOS`), the clustering knobs (`K_GRID`, `SEED_FRAC` for the
labelled seed, `KNN_K`), and all paths. Everything is overridable by env var so an
eval shrinks into one foreground window (see [Running](#8-running)).

### `data.py` — traces, sparse repositories, OOD, and the confound audit

`load_trace_pool()` builds the balanced **pool**. **Positives** take each
Attack_600 attack's `multi_turn_queries` (the ~5 ActorAttack sub-queries) and
**join them into one trace string** — a single distributed-misuse campaign member,
label 1, grouped by `query_id`. **Negatives** render each UltraChat conversation's
user turns into one trace the same way (label 0). The pool holds ≥500 of each.
`sample_repository(pool, size, base_rate)` then draws a repository with a
`base_rate` fraction positive — the sparse (~5%) Meerkat setting, or 0.5 for the
balanced control. `load_ood_cstm()` renders each CSTM-Bench scenario's sessions
into one trace (label 1 iff `scenario_class == "attack"`) — the **real**
out-of-distribution repository. `confound_report()` is the honesty check: can raw
**trace character-length** alone separate the classes (`length_auc`)? Mean-pooled
embeddings mitigate a length tell, but we measure and report it regardless.

### `cluster.py` — the embedder, k-means, and the three localizers

`get_embedder()` lazily loads bge (or minilm) **once** and returns
`embed_text(str) -> vec`; `embed_traces()` embeds a list and **L2-normalizes**
rows (so k-means uses cosine geometry). `choose_k()` runs k-means for each k in
`K_GRID` and picks the best **silhouette**; `kmeans_labels()` returns the cluster
id per trace. The three localizers share the `fit(emb, seed_idx, seed_labels)` /
`score(emb)` API: `PerTraceMonitor` fits logistic regression on the seed
embeddings (no cross-trace structure); `KMeansEnrich` clusters the **whole** repo
and scores each trace by its cluster's seed-estimated, Laplace-smoothed violation
fraction; `KnnPurity` scores each trace by the attack fraction among its `KNN_K`
nearest seed-labelled traces. `average_precision`, `roc_auc`, and
`cluster_purity` are the metrics. A CPU self-test on **synthetic** embeddings — a
tight 5% "attack" blob far from a diffuse benign cloud — asserts `KMeansEnrich`
and `KnnPurity` reach AP well above `PerTraceMonitor` at that sparse rate, with no
model load.

### `run_meerkat.py` — the orchestrator

`main()` embeds the pool once (caching to `EMB_CACHE`), runs the `confound_report`,
then for each base rate in {0.05 sparse, 0.5 balanced} builds `N_REPOS`
repositories, fits each method on a `SEED_FRAC` labelled seed, scores the rest, and
**averages AP + ROC-AUC over repositories** (mean + bootstrap CI). It records
**cluster quality** (best k, the silhouette grid, cluster purity, and the
campaign's max-cluster recall — does the campaign really concentrate in one
cluster?), then trains and evaluates on **CSTM-Bench** for the OOD numbers. Writes
`results.json` (schema in the contract) **before** the summary print, and renders
three PNGs: a 2-D PCA scatter with the attack cluster highlighted, AP-by-method
across base rate, and silhouette vs. k.

### `infer.py` — per-trace vs. cluster, side by side

Builds one small sparse repository (`MK_INFER_REPO`, default 200 traces at the 5%
rate), embeds and clusters it, fits `PerTraceMonitor` and `KMeansEnrich` on a small
seed, and prints two lists: the **per-trace monitor's top-scored traces** (mostly
benign false positives — the campaign is diluted) and the **k-means cluster the
attack campaign concentrates in** (its size, purity, and campaign-recall), followed
by cluster-enrichment's top traces and the two methods' Average Precision. All
model-touching code is under `main()`.

---

## 6. The dataset

Real attack decompositions rendered as traces, diluted into benign traffic at the
paper's sparse rate, with a real OOD benchmark on top.

| role | dataset (loader) | what it is | label |
|---|---|---|---|
| **positives** | **SafeMTData** `Attack_600` (`data.py`) | each ActorAttack attack's ~5 `multi_turn_queries` **joined into one trace**; `query_id` groups | **1** (violation) |
| **negatives** | **UltraChat 200k** `HuggingFaceH4/ultrachat_200k` (`data.py`) | each benign conversation's user turns joined into one trace | **0** (benign) |
| **the repository** | sampled from the pool (`data.sample_repository`) | `REPO_SIZE` traces at `BASE_RATE` (sparse **5%**, or 0.5 balanced) | mixed |
| **OOD** | **CSTM-Bench** `intrinsec-ai/cstm-bench` (`data.py`) | real cross-session scenarios; `attack` vs benign, each a set of sessions rendered to a trace | **1** iff attack |

**Positives — Attack_600 decompositions as traces.** SafeMTData's `Attack_600` is
600 multi-turn attacks from ActorAttack/Derail-Yourself (Ren et al. 2024,
arXiv:2410.10700), which decomposes a harmful goal into a chain of benign-seeming
sub-queries. We render each attack's sub-queries into **one trace string** (the
agent's event sequence), so a repository of these traces is exactly a
distributed-misuse **campaign**: many members that share an escalation shape. That
shared shape is what makes them cluster.

**The ≥500/class pool and the sparse 5% sampling.** Per the project's hard data
rubric, `load_trace_pool()` holds **≥500 attack and ≥500 benign** traces. The
Meerkat setting is not a balanced test set, though — it is a **needle-in-haystack**
one. `sample_repository()` therefore draws repositories at `BASE_RATE = 0.05`: a
few dozen campaign traces among several hundred benign ones. We report **Average
Precision** at that sparse rate (the metric that rewards ranking the rare
positives high) and **ROC-AUC** at a balanced 0.5 rate as a sanity control. Every
headline averages over `N_REPOS` independently sampled repositories with bootstrap
CIs, so the number is not one lucky draw.

**CSTM-Bench — the real OOD benchmark.** `intrinsec-ai/cstm-bench` is the released
cross-session-threat benchmark (Cross-Session Threats, arXiv:2604.21131): genuine
multi-session scenarios (attack vs benign), small (~52 attack + ~56 benign). We
build the repository, cluster, seed-label a few CSTM traces, and report AP/ROC-AUC
on it with **no** in-distribution retuning — an honest transfer number.

**The confound audit — and why it matters.** The obvious shortcut is **length**:
if joined attack traces are systematically longer than benign ones, a model could
separate the classes on character count alone and clustering would earn no credit.
`confound_report()` measures the AUC of raw **trace character-length** against the
label (`length_auc`). ≈0.5 means no trivial tell and a high localizer AP is real
cluster structure; well above 0.5 means the headline must be discounted for a
length artifact. The numbers are recorded in `results.json` and quoted beside the
method APs in [Section 9](#9-results--measured-vs-the-claim).

---

## 7. The cross-trajectory series

This lesson is the **unsupervised, repository-scale** member of a four-lesson arc
that reads the same idea — *a threat that no single trace carries is visible only
when you look at many traces together* — at growing scope:

| | `multiturn_jailbreak` | `trajguard` | `cross_trajectory` | `meerkat` (this lesson) |
|---|---|---|---|---|
| the unit | a conversation **turn** | a generated **token** | an agent **trajectory** | a whole agent **trace** |
| the collection | one ordered chat | one ordered generation | one **set** of K agents | a **repository** of many traces |
| the attack | Crescendo / ActorAttack escalation | a completion drifting to harm | a goal decomposed across K agents | a **sparse (~5%) distributed** campaign |
| the method | GRU / attention over turns | sliding window over tokens | learned set-pooling | **unsupervised clustering + enrichment** |
| supervision | supervised | supervised | supervised | **semi-supervised (a few seed labels)** |

`cross_trajectory` **learns** an aggregator over one sample's K trajectories;
`meerkat` **clusters** an unlabelled repository of many traces and needs only a
handful of seed labels to name the enriched cluster. Same instinct — read the set,
not the trace — one supervised and per-sample, the other unsupervised and
per-repository.

---

## 8. Running

From the **repo root** (`steeringresearch/`):

```bash
# CPU-only self-tests (NO model, NO big download):
python -m steering_tutorials.meerkat.cluster   # 3 localizers on synthetic embeddings
python -m steering_tutorials.meerkat.data       # small pool load + confound report

# The full pool -> embed -> sparse/balanced repositories -> OOD run
# (needs the bge-base-en-v1.5 embedder, ~0.4 GB):
python -m steering_tutorials.meerkat.run_meerkat

# Watch a sparse campaign hide from per-trace and get surfaced by clustering:
python -m steering_tutorials.meerkat.infer
```

**Env caps** (shrink an eval into one foreground window — the host's RAM, not
VRAM, is the wall):

| var | meaning | default |
|---|---|---|
| `MK_N_ATTACK` | attack traces in the pool | 500 |
| `MK_N_BENIGN` | benign traces in the pool | 500 |
| `MK_BASE_RATE` | sparse attack fraction per repository | 0.05 |
| `MK_EMBED` | embedder: `bge` or `minilm` | `bge` |
| `MK_INFER_REPO` | traces in `infer.py`'s demo repository | 200 |

```bash
# a fast smoke on the faster minilm embedder:
MK_EMBED=minilm MK_N_ATTACK=120 MK_N_BENIGN=120 \
  python -m steering_tutorials.meerkat.run_meerkat
```

On Windows PowerShell set env vars first, e.g. `$env:MK_EMBED = "minilm"`.

**No judge.** This is a **detection** lesson: a localizer reads a signal off frozen
trace embeddings, exactly as in lesson 1. There is no generation and no LLM judge —
so the off-family-judge discipline of the steering lessons does not apply here
(`results.json` records no judge).

---

## 9. Results — measured vs. the claim

**[PENDING GPU RUN]** — the harness, self-tests, and this section's structure are
in place; the numbers below are filled from `artifacts/results.json` once the
bge-embedder run completes. The falsifier and reading are pre-registered here
**before** the run.

**The claim under test.** Safety violations in agent repositories are sparse
(~5%), distributed, and adversarially diluted, so a per-trace monitor that scores
each trace alone misses them; **clustering the repository** concentrates the
campaign into one cluster and cluster-enrichment scoring surfaces it, reaching
Average Precision far above the per-trace baseline (Meerkat, arXiv:2604.11806).

Averaged over `N_REPOS` sampled repositories, bge-base-en-v1.5 embeddings, a
`SEED_FRAC` labelled seed, bootstrap 95% CIs. The **sparse (5%)** AP column is the
one that matters (balanced is the sanity control where per-trace can already do
fine).

| method | Sparse (5%) **AP** | Sparse ROC-AUC | Balanced ROC-AUC |
|---|---|---|---|
| `per_trace` (diluted-campaign-defeated baseline) | _pending_ | _pending_ | _pending_ |
| `kmeans_enrich` (Meerkat-style — headline) | _pending_ | _pending_ | _pending_ |
| `knn_purity` (clustering-free density proxy) | _pending_ | _pending_ | _pending_ |

OOD transfer — seed-labelled on a few CSTM-Bench traces, scored on the rest with no
in-distribution retuning:

| method | CSTM-Bench AP | CSTM-Bench ROC-AUC |
|---|---|---|
| `per_trace` | _pending_ | _pending_ |
| `kmeans_enrich` | _pending_ | _pending_ |
| `knn_purity` | _pending_ | _pending_ |

**Cluster quality** (sparse repo, from `results.json`): `best_k` _pending_,
`cluster_purity` _pending_, `campaign_max_cluster_recall` _pending_ — the fraction
of the campaign captured by its single most-enriched cluster (the mechanism the
whole claim rests on). **Confound audit:** `length_auc` _pending_ — we quote it
beside the method APs and claim only the margin **above** a length tell, never the
raw gap over `per_trace`.

**Pre-registered falsifier.** The thesis is the ordering `AP(kmeans_enrich) >
AP(per_trace)` at the **5% sparse** base rate. If `kmeans_enrich` comes back
**≤ `per_trace`** on sparse AP, then "clustering surfaces the distributed
campaign" is **FALSE** for this setup and must be reported as a negative result —
no reclassifying to the balanced regime to rescue it, no moving the goalposts to a
richer base rate.

---

## 10. Honest caveats

- **Screening tier, not evaluation.** One embedder, one seed fraction, a few
  hundred traces per repository — a directional demo, not the n ≥ 7 seeds + rigor
  contract CLAUDE.md reserves the word "winner" for. Do not over-read the ordering.
- **Our localizer is NOT Meerkat's agentic-search system.** The paper's method
  pairs clustering with an **agentic search** step that reasons over clusters and
  describes violations in natural language, discovering failures with no seed
  scenarios. We implement only the **cluster-density localizer** (k-means +
  enrichment scoring from a small seed). It is faithful to the *clustering
  intuition* — the shape of the cloud carries the campaign — but it is **not** the
  paper's reasoning agent, and it is semi-supervised where the paper is closer to
  unsupervised.
- **Constructed traces are NOT live agent logs.** Positives render an ActorAttack
  attack's sub-queries into one trace *as if* it were a logged agent session;
  benign traces are UltraChat conversations. We do **not** have real logs of a
  deployed fleet running a coordinated campaign. The lesson tests whether the
  *latent structure* of such a campaign is recoverable by clustering, not whether a
  production system emits exactly these traces.
- **Adaptive attacks can DISPERSE the cluster.** The whole method rests on campaign
  traces being *similar* enough to cluster. An adversary who deliberately
  **varies** each session's surface form — different actors, phrasings, orderings —
  spreads the campaign across many clusters and defeats enrichment scoring. This is
  the paper's own acknowledged weakness, and it is why clustering is a *detector*,
  not a guarantee.
- **CSTM-Bench is small and OOD.** ~52 attack + ~56 benign scenarios, a different
  distribution from our constructed pool — a transfer probe with wide CIs, not a
  precise evaluation. Read the OOD number as directional.
- **Inspired-by, not a paper reproduction.** The architecture (bge embeddings +
  k-means + cluster-enrichment) operationalizes the *clustering idea* of Meerkat;
  it is **not** a faithful reimplementation of the paper's full agentic-search
  system, and one cited id is marked `[UNVERIFIED]` pending a WebFetch check (see
  `AUDIT.md`).

---

## 11. Repository

Source and full artifacts:
<https://github.com/dlmastery/steeringresearch/tree/master/steering_tutorials/meerkat>

Cited: Meerkat / Detecting Safety Violations Across Many Agent Traces
(arXiv:2604.11806), CSTM-Bench / Cross-Session Threats (arXiv:2604.21131),
GroupGuard (arXiv:2603.13940), BGE / C-Pack (arXiv:2309.07597 `[UNVERIFIED]`);
positives from ActorAttack / Derail Yourself (arXiv:2410.10700).

See also
[the course map](../README.md),
[the supervised sibling — cross_trajectory](../cross_trajectory/README.md)
(a learned set-aggregator over K agents),
[the turn-level lesson — multiturn_jailbreak](../multiturn_jailbreak/README.md),
[the token-level lesson — trajguard](../trajguard/README.md), and
[lesson 1 — the single-prompt activation probe (READ)](../hello_world/README.md),
whose activation-reading idea the whole series generalizes.
