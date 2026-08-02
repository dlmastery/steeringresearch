# SYNTH 2026-08 — Selection, Deduplication, and Curation of Synthetic Training Data with Embeddings

**Scan date:** 2026-08-02
**Hard date filter:** arXiv ID prefix **2606 / 2607 / 2608 only**. Nothing 2605 or earlier is cited.
Older canonical work (SemDeDup arXiv:2303.09540, Coverage-centric Coreset Selection arXiv:2210.15809,
Moderate Coreset, Forgetting scores, EL2N, LESS, DataComp-LM) is **named as lineage only** and is
deliberately NOT cited as evidence here.
**Verification:** every arXiv id below was WebFetched and its title + author list confirmed against the
arXiv abstract page. No `[UNVERIFIED]` entries — all fetches succeeded.
**Numbers discipline:** every number in this document was read from the fetched paper page. Where a
paper reports no numbers on the surfaces fetched, this document says so rather than inventing one.

---

## 0. Summary table

| Paper | arXiv id | Date | Method (one line) | vs random @ matched budget? | Encoder-bias acknowledged? | Laptop-feasible? |
|---|---|---|---|---|---|---|
| Coresets Before Score Sets | 2607.09739 | 2026-07-02 | Facility-location over prompt embeddings picks a tiny benchmark subset that preserves LLM scores | **YES, numeric** (MRE 0.050 vs 0.084 at k=80) | **YES, explicitly** — argues text-only embeddings are model-independent *by design* | **YES** (embed once + greedy FL; sparsified to O(αn²/k) memory) |
| Data Pruning: Redundant, Problematic, and Interdependent Samples | 2606.21916 | 2026-06-20 | Audits Forgetting + EL2N pruning; shows scores are relational, not intrinsic | **YES, and random WINS at high pruning rates** | Partially — score depends on the scoring model's own training run | **YES** (2-layer MLP / VGG-16 scale) |
| Few-Medoids | 2607.05891 | 2026-07-07 | Pick the k samples nearest each class centroid in the teacher's latent space | **YES, numeric** (e.g. 28.21% vs 18.19% on Food-101 k=16) | **NO** — selects in teacher feature space, does not test a different encoder | **YES** (Euclidean distances only) |
| When Sample Selection Bias Precipitates Model Collapse | 2606.13732 | 2026-06-11 | Proves a verifier with a *biased local reference* accelerates collapse; fixes it with Wasserstein proxy references | **YES, numeric** (FID 71 vs random 106, CIFAR-10) | **YES in substance** — the whole paper is about the selector's reference distribution rigging the outcome | Partial (DDPM training is GPU-heavy; the selection logic is cheap) |
| emb-diversity | 2607.19848 | 2026-07-22 | Open tool implementing 22 embedding-based diversity measures over any HF encoder | No (measurement tool, not a selector) | **NO** — states measures "work with any embedding model" but runs **no** encoder-sensitivity analysis | Partial — O(n²) pairwise; some measures OOM at 100k |
| Cross-Attention Calibrated Deduplication (CACD) | 2607.24332 | 2026-07-27 | Cross-encoder + attention-entropy "New Information Score" replaces cosine thresholding for RAG chunk dedup | No | Implicitly — argues the pooled-vector encoder is the failure point | **YES** (51.0 s/config, 7x faster than their cosine baseline) |
| When Hard Negatives Hurt | 2606.01304 | v1 2026-05-31, 2606 announce | Names the generative-discriminative gap: fluent LLM negatives ≠ boundary-violating negatives | Not on this axis | Partially — critiques retriever-score-based mining as self-referential | Partial (retriever training) |
| CuratorKIT | 2606.21631 | 2026-06-19 | Open pipeline: Exact (SHA-256) + MinHash-LSH + FAISS EmbeddingDeduplicator with persistent cross-run index | No (engineering artifact, no benchmark) | No | **YES** (FAISS optional, disk-backed index) |
| FOLD / RAD | 2606.03001 | 2026-06-02 | Online fuzzy dedup via incrementally-updated HNSW with bitmap-corrected Jaccard | No (recall vs LSH, not vs random) | N/A (Jaccard, not embeddings) | **YES, streaming** — never needs the full corpus resident |
| Internal Data Repetition Destroys LMs | 2606.24998 | 2026-06-23 | Repetition damage is non-monotonic in (subset size x repeat count) | No | No | No (pretraining scale, 344M) |
| Post-Generation Curation via HO/HE Splitting | 2607.02637 | 2026-07-02 | Split each real class into canonical (homogeneous) + non-redundant (heterogeneous), score synthetics on fidelity minus canonical redundancy | Not stated (vs "SOTA data selection baselines") | No | Partial (image generators) |
| KITE | 2607.17043 | 2026-07-19 | Kernel Boundary Uncertainty (RBF over answer embeddings, Rényi-2 entropy) keeps the 0.2–0.8 uncertainty band | **NO — explicit gap** | **YES, in Limitations** — acknowledges the external verifier is a distillation channel | Partial (needs gpt-5-mini verifier) |
| SCOSS | 2607.09100 | 2026-07-10 | Score-stratified sampling + ensemble aggregation over independently sampled subsets | Yes (qualitatively; random + class-balanced random are baselines) | No | **YES** |
| Submodular Information Measures: Variance and Separation | 2607.27660 | 2026-07-30 | Theory: Graph-Cut TI = within-class variance, LogDet TI = generalized variance, Facility-Location TI = imbalance-aware separation | No (theory + synthetic) | No | **YES** (theory) |

---

## 1. Coresets Before Score Sets — arXiv:2607.09739

**Title:** *Coresets Before Score Sets: Evaluation-Unsupervised Prompt Subset Selection for LLM Benchmarks*
**Authors:** Jihan Yao, Gantavya Bhatt, Arnav Das, Peter Jin, Ke Bao, Qiaolin Yu, Khushi Bhardwaj, Chang Su,
Jialei Wang, Yikai Zhu, Sugam Devare, Damon Mosk-Aoyama, Zhen Dong, Venkat Krishna Srinivasan,
Yineng Zhang, Oleksii Kuchaiev, Jiantao Jiao, Banghua Zhu, Jeff Bilmes
**Submitted:** 2 July 2026

**Method.** Select a prompt subset *before* any model has been scored ("evaluation-unsupervised"), so the
coreset does not depend on which models happen to be available. Each prompt is concatenated with its
ground-truth metadata and embedded: `e_i = LLM(text_i)` using **Qwen3-Embedding-4B**. A **facility
location (FL)** submodular function is then greedily maximized over those embeddings. FL is compared
against determinantal point processes, submodular mutual information variants, and score-based
selectors — twelve baselines in total.

**Numbers read.**
- Pool: **61,498 prompts**. Budgets **k ∈ [70, 200]** (roughly 0.11%–0.33% retention).
- **k=80:** FL mean relative error **0.050 ± 0.015** vs **random 0.084 ± 0.005** (lower is better).
- **k=200:** FL **0.053 ± 0.001** vs **random 0.079 ± 0.004**.
- Sparsification reduces memory from **O(n²) to O(αn²/k)**. No explicit wall-clock runtime given in the
  main text; FL is stated to be "substantially cheaper to compute" than DPP baselines.

**Encoder-bias position — the strongest in this scan.** The paper confronts the question head-on and
takes the *opposite* side from most of the field. It rejects IRT-derived (model-response-derived)
embeddings precisely because they depend on the model pool, and argues: *"Semantic embeddings are
computed from sample text alone and are therefore identical regardless of which models are available,
making them more reliable for facility location."* Note carefully what this does and does not buy:
it removes dependence on the **models being scored**, but it does NOT remove dependence on the
**embedding model doing the selecting** — Qwen3-Embedding-4B's own blind spots still decide what
survives. The paper does not test that second-order dependence.

**What this project could use.** This is the closest published template to what a policy-guard eval
needs. It is a direct, implementable recipe: embed once with a *held-out* encoder, run greedy FL,
compare to random at matched k, report MRE. The sparsified O(αn²/k) formulation is what makes it
laptop-feasible at ~1000-label scale (trivially so — 61k prompts is already 60x our scale). **The
adaptation this project must make:** select with a *third-party* encoder (Qwen3-Embedding), never with
EmbeddingGemma-300M or MiniLM, since those are the encoders under test. This paper's own logic
demands that adaptation even though the paper does not state it.

---

## 2. Data Pruning: Redundant, Problematic, and Interdependent Samples — arXiv:2606.21916

**Title:** *Data Pruning: Redundant, Problematic, and Interdependent Samples*
**Authors:** Leon Freese, Marthinus W. Theunissen
**Submitted:** 20 June 2026

**Method.** A deliberately negative audit rather than a new method. Two established difficulty scorers —
**Sample Forgetting Score** and **Error L2-Norm (EL2N)** — are re-run under noiseless and 30%-label-noise
conditions on a synthetic set (10,000 samples, 2-layer MLP), MNIST (55,000, 2-layer MLP), and CIFAR-10
(45,000, VGG-16).

**Numbers read.**
- Noiseless, at moderate pruning: synthetic **~40% prunable** with **~2% accuracy** gain over random;
  MNIST **~90% prunable** with minimal gain; CIFAR-10 **~40% prunable** with **~2% accuracy** over random.
- **At extreme pruning levels the random baseline outperforms both scoring methods** — a direct
  contradiction of their design intent.
- Under **30% label noise** both methods "fail completely." **Reversing the ranking order** produces
  "drastic improvements," with reversed-EL2N reaching performance "similar to an Oracle ranking."
- Central conclusion: *"a sample's score depends on the presence of other samples in the dataset"* —
  importance is relational, not intrinsic.

**What this project could use.** Three things, all defensive.
1. **The measured gain over random is ~2 accuracy points, not a transformation.** Any claim this program
   makes about a clever selector must be priced against that: 2 pp is inside many seed bands.
2. **Random wins at aggressive budgets.** If the guard's training budget is small (which at ~1000 labels
   it is), the prior should be that random is competitive, and the burden is on the selector.
3. **Interdependence kills naive dedup.** If a sample's score depends on which other samples are present,
   then removing a "duplicate" changes the score of everything left. This is the formal statement of the
   failure mode this project is worried about: dedup and difficulty-scoring are not commutative, and a
   pipeline that dedups first and scores second is measuring a different dataset than the one it scored.

**Falsifier this suggests for our harness:** run the guard selection pipeline forward and with the
difficulty ranking *reversed*. If reversed does not lose, the scorer is not measuring difficulty.

---

## 3. Few-Medoids — arXiv:2607.05891

**Title:** *Few-Medoids: An Embarrassingly Simple Coreset Selection Method for Few-Shot Knowledge Distillation*
**Authors:** Cemil-Andrei Dilmac, Florinel-Alin Croitoru, Radu Tudor Ionescu
**Submitted:** 7 July 2026

**Method.** Select, per class, the k samples nearest the class centroid (mean representation) in the
**teacher network's latent feature space** `f_φ`. Pure Euclidean distance; no training, no gradients.

**Numbers read** (ResNet-34 → ResNet-18 distillation, Few-Medoids vs Random):

| Dataset | k=16 | k=32 |
|---|---|---|
| CIFAR-10 | **34.31%** vs 29.98% | **41.68%** vs 36.18% |
| CIFAR-100 | **29.67%** vs 24.66% | **41.99%** vs 40.27% |
| Food-101 | **28.21%** vs 18.19% | **43.36%** vs 33.42% |

**Encoder bias — NOT acknowledged.** This is the clearest un-acknowledged instance in the scan. Selection
happens in the teacher's feature space; the student is then trained on that selection. The paper does not
discuss whether choosing samples the teacher finds *canonical* systematically removes the samples the
student would find informative, and it does not run the ablation of selecting in a third network's space.

**Note on the gains.** The margins are large (Food-101 k=16: +10.02 pp) but so is the regime: k=16 per
class is far below the ≥500/class floor this project's rubric mandates. At k=32 the CIFAR-100 margin is
already down to +1.72 pp. Read this as evidence that **medoid selection matters most when the budget is
tiny** and decays as budget grows — which is the opposite of our operating regime.

**What this project could use.** The method is ~10 lines and free, so it is a legitimate cheap arm in a
selection bake-off. But its own numbers predict the gain shrinks at our budget, and its encoder-bias
posture is exactly the one this project must not adopt.

---

## 4. When Sample Selection Bias Precipitates Model Collapse — arXiv:2606.13732

**Title:** *When Sample Selection Bias Precipitates Model Collapse*
**Authors:** Xinbao Qiao, Xianglong Du, Wei Liu, Jingqi Zhang, Peihua Mai, Meng Zhang, Yan Pang
**Submitted:** 11 June 2026 (v1); revised 2 July 2026 (v2)

**The central claim, and it is the one this project cares about most.** Data selection is normally sold as
a remedy for synthetic-data collapse. This paper shows it can be the *cause*. When the verifier/selector
sees only a biased slice of the distribution, selection *"preferentially retains samples aligned with the
local manifold while pruning globally relevant tail modes."* The authors **theoretically prove that siloed
selection accelerates collapse and induces power-law diversity decay**, then fix it with **Wasserstein
proxy references** built across data silos without sharing raw data.

**Numbers read** (DDPM, 10 iterations; Table 2):

| Dataset | Random | Scheme II (Collab. Barycenter) | Scheme I (Collab. Geodesic) |
|---|---|---|---|
| CIFAR-10 (ExDir(1,0.1)) | FID 106 / P 0.53 / R 0.48 | FID 85 / P 0.57 / R 0.57 | **FID 71 / P 0.60 / R 0.58** |
| STL-10 | FID 95 / P 0.49 / R 0.53 | FID 69 / P 0.57 / R 0.63 | **FID 65 / P 0.66 / R 0.71** |
| CelebA | FID 96 / P 0.51 / R 0.28 | FID 75 / P 0.70 / R 0.62 | **FID 69 / P 0.69 / R 0.71** |

Figure 4 (Left) shows class-proportion collapse: with only "Airplane" as the local reference, that class
proportion degrades dramatically across iterations.

**Encoder identity.** The paper uses **VGG11** for selection scoring. It does **not** explicitly state
whether that scorer's architecture matches the generative model being evaluated — so on the narrow
"same encoder" question it is silent, even while being the paper that most directly establishes the
underlying mechanism.

**What this project could use.** This is the mechanistic citation for the bias worry in the brief. It
converts "filtering with the encoder under test rigs the evaluation" from an intuition into a proven
statement: *a selector whose reference distribution is narrower than the target distribution prunes tail
modes and drives power-law diversity decay.* An EmbeddingGemma-300M-filtered corpus evaluated with
EmbeddingGemma-300M is exactly the siloed-reference configuration this paper proves is degenerate. The
proposed fix — **aggregate multiple reference distributions rather than trusting one** — translates
directly to: select with an *ensemble* of encoders (or a disjoint one), never with the encoder under test.

---

## 5. emb-diversity — arXiv:2607.19848

**Title:** *emb-diversity: A Tool for Embedding-Based Measurement of Data Diversity*
**Authors:** Cantao Su, Menan Velayuthan, Esther Ploeger, Dong Nguyen, Anna Wegmann
**Submitted:** 22 July 2026

**Method.** An open tool implementing **22 diversity measures** over arbitrary Hugging Face encoders:
- *Distance-matrix based:* Mean Pairwise Distance, Sum Pairwise Distance, Energy, Span (Medoid), KNN,
  Chamfer Distance, Sum Bottleneck, Bottleneck, Sum Diameter, Diameter
- *Distance-graph based:* MST Dispersion, Graph Entropy, HamDiv
- *Kernel-matrix based:* Vendi Score, Rényi Kernel Entropy, Log-Determinant, DCScore
- *Other:* Bins Entropy, Convex Hull (3D), Cluster Inertia, Span (Centroid), Geometric Mean Std

Three defaults: **Mean Pairwise Distance, Vendi Score, Graph Entropy**. Pre-defined "diversity axes" for
semantics and style, with default models `all-mpnet-base-v2` and `AnnaWegmann/Style-Embedding`.

**Numbers / cost read.** Most measures need **O(n²) pairwise distances**. Per Figure 5, Hamiltonian
diversity is *"prohibitively expensive"* at 10k samples; Vendi Score *"scales well"*; some measures **run
out of memory at 100k** instances. Appendix F reports "strong correlations between many diversity
measures" across **17 Common Pile domains**, with weaker correlations for Convex Hull Volume and Diameter.

**Encoder bias — NOT acknowledged, and this is a notable gap.** The paper's selling point is that the
measures *"work with any embedding model and any data that can be embedded."* It runs case studies with
different encoders for different axes (semantic vs style) but performs **no systematic analysis of whether
the diversity ranking of two corpora is stable across encoder choices.** Flexibility is presented as an
unqualified virtue rather than as a free parameter that could flip a conclusion.

**What this project could use.** Practically the most immediately usable artifact in the scan: it is a
tool, it runs on CPU, and Vendi Score is explicitly stated to scale. At ~1000 labels, O(n²) is 10^6
pairs — trivial. **The experiment this paper leaves on the table is exactly our question:** compute the
same diversity measure on the same corpus under EmbeddingGemma-300M vs MiniLM vs mpnet and report the
rank correlation. If the ranking is unstable, every embedding-based curation claim in this program
(ours included) inherits an unpriced free parameter.

---

## 6. Cross-Attention Calibrated Deduplication (CACD) — arXiv:2607.24332

**Title:** *Cross-Attention Calibrated Deduplication for Retrieval-Augmented Generation System*
**Authors:** Phuong Le Huy, Nam H. Nguyen, Quan V. Dang
**Submitted:** 27 July 2026

**Method.** The one paper in this scan that directly attacks the **"duplicate vs hard pair"** distinction.
It replaces cosine thresholding over pooled vectors with a **cross-encoder** that preserves token-level
granularity, plus a **New Information Score (NIS)** derived from **attention entropy** measuring *"how
much of a chunk is not explained by a candidate already kept,"* plus majority voting across candidates.

**The criticism it levels at cosine thresholding, quoted:** *"a single vector can lose the fine-grained,
token-level detail needed to tell a true duplicate apart from a chunk that just shares the same topic."*
This is precisely the failure mode in the brief, stated in RAG terms: pooled-cosine dedup cannot separate
*redundant* from *topically adjacent but informationally distinct*.

**Numbers read.**
- Deduplication rate: **9.75%** average chunk removal.
- Speed: **51.0 s per configuration**, **27% faster** than the NERExact baseline (69.6 s) and **7x faster**
  than cosine-similarity (356.7 s).
- **Authors' own caveat, quoted:** results derive *"from a single dataset, so we present them as an early
  comparison, not a general claim."*

**What this project could use.** The NIS formulation — score a candidate by *residual* information given
what is already kept, not by pairwise similarity to it — is directly portable to guard-policy dedup and is
the correct shape for the hard-pair problem: a cross-boundary near-duplicate has high similarity but also
high residual information (it carries the label flip), so NIS keeps it where cosine deletes it. Cost is
the obstacle: a cross-encoder is O(n²) forward passes, not O(n²) dot products. At ~1000 examples that is
still feasible on a laptop; at corpus scale it is not. The honest single-dataset caveat should be carried
forward with any citation.

---

## 7. When Hard Negatives Hurt — arXiv:2606.01304

**Title:** *When Hard Negatives Hurt: Bridging the Generative-Discriminative Gap in Hard Negative Synthesis for Retrieval*
**Authors:** Zhicheng Zhang, Jiwei Tang, Kuicai Dong, Xiaopeng Li, Jieming Zhu, Jingyu Li, Qianhui Zhu,
Fengyuan Lu, Wang Jiaheng, Gang Wang, Hai-Tao Zheng, Zhaocheng Du
**Dates:** v1 31 May 2026, revised 7 Jun 2026. **DATE FLAG:** the arXiv id prefix is **2606** (June
announcement) but v1 predates the 2026-06-02 cutoff by two days. Included on the ID-prefix rule; flagged
so the caller can drop it if the stricter date rule governs. Accepted at KDD 2026.

**Content.** Names the **generative-discriminative gap**: *"LLM generation optimizes for fluent, plausible
text, while contrastive learning demands strategic violations of relevance at the decision boundary."*
Naively adding generated negatives to contrastive training degrades retrieval. It also states the
structural indictment of mined negatives that matters here: mined negatives are *"bounded by corpus
availability, selected by retriever score rather than diagnostic value, and increasingly contaminated by
false positives as the retriever improves."*

**No specific nDCG numbers were available on the abstract page fetched.** Do not quote numbers from this
paper without reading the full text.

**What this project could use.** The middle clause is the encoder-bias argument in retrieval clothing:
*selected by retriever score rather than diagnostic value*. Mining hard negatives with the retriever under
test selects for what that retriever already gets wrong in a self-referential loop, and the contamination
gets **worse** as the retriever improves. That is the same trap as filtering with the encoder under test,
and it is the sharpest available citation for the claim that the bias is not a fixed offset but grows with
model quality — meaning a stronger EmbeddingGemma makes the rigging worse, not better.

---

## 8. CuratorKIT — arXiv:2606.21631

**Title:** *CuratorKIT: Data Curation and Synthetic Data Generation for LLM Post-Training*
**Authors:** Soham Bhattacharjee, Karun Sharma, Vinay Kumar Sankarapu, Pratinav Seth
**Submitted:** 19 June 2026

**Method.** An open-source Python library covering the full post-training data lifecycle: six source
format readers, data-hygiene layers, eight LLM-powered generation tasks, five training-ready export
formats, 100+ providers via LiteLLM, compatible with TRL / Unsloth / AlignTune. Three stacked dedup
stages:
1. **ExactDeduplicator** — SHA-256 over normalised text, applied before generation/export.
2. **MinHashDeduplicator** — character n-gram MinHash with **LSH banding** to avoid all-pairs comparison.
3. **EmbeddingDeduplicator** — persistent on-disk embedding index, **FAISS** ANN when available,
   **default cosine threshold 0.92**, and — the useful part — the index **persists across runs**, so
   samples accepted in an earlier run are not reintroduced when the pipeline is re-run on new data.

**Numbers read.** None. **No benchmark, no random-selection comparison, no downstream validation.** The
0.92 threshold is a configured default, not a tuned or ablated value — nothing in the paper establishes
that 0.92 is right for any corpus, and nothing measures what it deletes.

**What this project could use.** The **cross-run persistent index** is a genuinely good engineering idea
worth copying for iterative synthesis: without it, each generation round re-introduces content the last
round rejected. But treat the 0.92 default as what it is — an unvalidated constant. This paper is an
artifact citation, not an evidence citation, and it is a live example of the field shipping a
semantic-dedup threshold with no measurement of the hard examples it removes.

---

## 9. FOLD / RAD — arXiv:2606.03001

**Title:** *FOLD: Fuzzy Online Deduplication for Very Large Evolving Datasets via Approximate Nearest Neighbor Search*
**Authors:** Nelson Bore, Pritish Mishra, Constantin Adam, Eyal de Lara, Oana Balmau
**Submitted:** 2 June 2026 (v1); revised 11 June 2026 (v2). *Exactly on the cutoff date.*

**Method.** The system (RAD, Retrieval-Augmented Deduplication) maintains an **incrementally updated HNSW
index over admitted documents** instead of LSH. Its specific contribution is a **bitmap representation**
that improves Jaccard matching during ANN search, fixing *"distance score crowding"* — the effect where
standard HNSW loses recall because near-duplicate distances bunch together and become indistinguishable.

**Numbers read.** At **30M documents**: **0.94–0.97 recall** relative to state-of-the-art LSH solutions,
with **up to 8x throughput increase**.

**Caveat for this project:** it is **Jaccard/shingle-based, not embedding-based.** It does not address
semantic near-duplicates or the hard-pair problem at all.

**What this project could use.** The architecture, not the metric. **Streaming/online is the right shape
for a laptop:** an incrementally updated index never needs the full corpus resident, which directly
answers the brief's memory question. The distance-score-crowding diagnosis also transfers to cosine space
and is worth checking in ours — if EmbeddingGemma cosines for the guard corpus bunch in a narrow band,
any global threshold is separating noise, and that would be a measurable, judge-free defect.

---

## 10. Internal Data Repetition Destroys Language Models — arXiv:2606.24998

**Title:** *Internal Data Repetition Destroys Language Models*
**Authors:** Jessica Chudnovsky, Joshua Kazdan, Noam Levi, Rylan Schaeffer, Yegor Denisov-Blanch, Bo He,
Mehmet Donmez, Sanmi Koyejo, David Donoho
**Submitted:** 23 June 2026

**Finding.** Repetition damage is **non-monotonic**: *"repeating a moderately sized subset a moderate
number of times damages performance more than repeating a large subset a few times or a small subset many
times."* The worst case is interior to the (subset size x repeat count) grid, not at either extreme.

**Number read.** For a **344M-parameter** model with **10% of FLOPs** allocated to repeated data, the most
damaging repeat count produces losses equivalent to training without repetition on only **67% of available
FLOPs** — i.e. roughly a third of compute wasted.

The abstract does not specify the duplicate-detection method (the controlled experiments use exact-document
repetition); do not attribute an embedding-based detector to this paper.

**What this project could use.** It reframes the dedup threshold question. Because damage is non-monotonic,
"how aggressive should the threshold be?" has no monotone answer — a *partially* effective dedup pass can
land the corpus in the worst region of the grid, worse than either no dedup or thorough dedup. That is a
strong argument for measuring the *resulting repeat-count distribution* after dedup rather than reporting
only the fraction removed. Scale caveat: 344M pretraining is far from our ~1000-label fine-tuning regime,
so this is a mechanism to be aware of, not a transferable number.

---

## 11. Post-Generation Curation via Homogeneous-Heterogeneous Splitting — arXiv:2607.02637

**Title:** *Post-Generation Curation of Synthetic Images via Homogeneous-Heterogeneous Splitting*
**Authors:** Disheng Liu, Tuo Liang, Chaoda Song, Yu Yin
**Submitted:** 2 July 2026

**Method.** Split each **real** class into a canonical **Homogeneous (HO)** subset and a non-redundant
**Heterogeneous (HE)** subset, then score synthetic images with a **fidelity-diversity criterion that
rewards semantic alignment while penalizing canonical redundancy.** The paper explicitly notes that prior
clustering-based curation — using real-data cluster centroids as anchors to retrieve synthetic samples —
**risks reducing downstream diversity**, which is the mode-collapse-by-curation concern.

**Number read.** *"matches the real-data performance with up to 40% fewer synthetic samples."* Benefits
reported for both classification and segmentation. **No matched-budget random comparison stated** — the
claim is against "state-of-the-art data selection baselines."

**What this project could use.** The HO/HE split is a clean, portable idea and the correct structural
answer to the hard-pair problem: **do not score a candidate against the class as a whole; score it against
the class's canonical core, and reward what the core does not already cover.** For guard policies, the HO
set is the stereotyped phrasing of a policy violation and the HE set is the long tail; a synthetic example
that duplicates HO is worthless while one that extends HE is exactly what zero-shot transfer needs. Note
this is an image paper — the transfer to text guard policies is untested and should be labelled inspired-by,
not reproduction.

---

## 12. KITE — arXiv:2607.17043

**Title:** *Learning from Synthetic Data without Model Collapse in Iterative Instruction Tuning*
**Authors:** Xiaonan Luo, Yue Huang, Kehan Guo, Ping He, Chuan Zou, Ting Hua, Xiangliang Zhang
**Submitted:** 19 July 2026

**Method.** KITE (Knowledge-boundary Instruction Tuning via Exploration): failure-guided data generation
plus **boundary-aware uncertainty curation**. Answers are embedded `h_i = φ(y^(i)) ∈ R^d`; an **RBF kernel**
gives a semantic similarity matrix (continuous, *not* discrete clustering); **Kernel Boundary Uncertainty
(KBU)** is the **Rényi-2 entropy of a likelihood-weighted kernel matrix**. Candidates are kept when
uncertainty falls in the percentile band **(u_min, u_max) = (0.2, 0.8)** — targeting the semantic knowledge
boundary, neither too easy nor beyond the model's reach.

**Reframing of collapse worth carrying.** Collapse appears as *"polarization of competence, where synthetic
training reinforces already strong skills while further degrading weak ones"* — not uniform degradation.
A mean metric hides it.

**Random baseline — an explicit and acknowledged gap.** Baselines are Self-Instruct, few-shot synthesis,
CDS, and ToEdit. **There is no random-selection-at-matched-budget arm.** So the paper does not establish
that KBU adds anything beyond generic diversification. Reported outcome is qualitative: KITE *"yields more
stable improvement than strong synthetic-data baselines."*

**Encoder/verifier bias — acknowledged.** Uses **gpt-5-mini as an external verifier**, not the model being
trained. Limitations concede *"the answer channel still carries an element of distillation,"* arguing the
verifier is applied identically across baselines so the confound is controlled rather than KITE-specific.
That is the right disclosure discipline even if the confound is not eliminated.

**What this project could use.** The **(0.2, 0.8) uncertainty band** is a concrete, implementable rule that
is *not* a similarity threshold — it keeps boundary examples by construction rather than deleting them, so
it is structurally immune to the hard-pair failure mode. Kernel entropy over ~1000 embeddings is a trivial
CPU computation. But the missing random arm means this is an untested hypothesis at our budget, and it
should be entered into our harness *with* the random arm the paper omits.

---

## 13. SCOSS — arXiv:2607.09100

**Title:** *A Coreset Selection Framework with Ensemble Aggregation for Image Classification*
**Authors:** Pedro Rocha Dantas, Lucas Pascotti Valem
**Submitted:** 10 July 2026

**Method.** **SCOre-Stratified Selection (SCOSS)**: partition training data into intervals by a chosen
score, sample from each interval, then aggregate predictions across multiple independent runs on
separately sampled subsets. Baselines include **moderate and random selection, each in original and
class-balanced versions.**

**Numbers read.** None specific. Findings are qualitative: SCOSS is *"competitive with baselines, often
the best choice for SGC"*, with favourable accuracy/efficiency trade-offs; SGC with SCOSS outperforms SVMs
on fine-grained datasets with limited labels. No percentages at named pruning rates on the surfaces
fetched. The abstract does not specify which difficulty score is used ("a chosen score").

**What this project could use.** Two transferable design choices rather than a result: (a) **stratified
sampling across the score range beats taking the top of it** — this is the generic defence against
difficulty-scoring deleting either the easy core or the hard tail; (b) **class-balanced random is included
as a baseline**, which is the right control — plain random is a weak strawman when classes are imbalanced,
and a guard-policy corpus is imbalanced. Adopt the *baseline set*, not the method.

---

## 14. Understanding Submodular Information Measure Based Objectives — arXiv:2607.27660

**Title:** *Understanding Submodular Information Measure Based Objectives for Representation Learning:
A Variance and Separation Perspective*
**Authors:** Rishabh Iyer, Truong Pham, Anay Majee
**Submitted:** 30 July 2026

**Content.** Theory connecting Submodular Information Measures to representation-learning quantities:

*Total Information (TI):* Graph Cut TI recovers **within-class variance**; LogDet TI captures **generalized
variance / covariance volume**; **Facility Location TI induces "imbalance-aware separation that emphasizes
rare and confusable classes."**

*Mutual Information (MI):* Graph Cut MI relates to **centroid separation and Fisher discrimination**;
LogDet MI captures **covariance-aware separation via Mahalanobis distance**; Facility Location MI measures
**nearest-mode representational overlap.**

Validated on controlled synthetic experiments independently varying variance, covariance, class imbalance,
class separation, and multimodal overlap. **No numerical comparisons vs random and no benchmark numbers**
on the surfaces fetched.

**What this project could use.** It supplies the *reason* facility location is the right submodular choice
in paper #1 rather than an empirical accident: **FL emphasizes rare and confusable classes.** "Confusable"
is the property we want preserved — the cross-boundary hard pairs. This is the theoretical bridge between
"FL beat random by 0.034 MRE" and "FL is the objective that structurally does not delete hard examples,"
and it is the citation to pair with 2607.09739 when justifying an FL-based selector.

---

## 15. Cross-cutting reading of the scan

**On the hard-pair problem (brief item 1).** Only **CACD (2607.24332)** attacks it directly and it does so
in RAG chunking, on one dataset, by the authors' own admission. **KITE (2607.17043)** avoids the problem
structurally by selecting on an uncertainty band rather than a similarity threshold. **2607.02637** avoids
it by scoring residual coverage against a canonical core. **CuratorKIT (2606.21631)** exemplifies the
problem: a shipped 0.92 default with no measurement of what it deletes. Nobody in this window published a
paper whose primary contribution is "we measured which hard examples semantic dedup removes."

**On matched-budget random comparison (brief item 2).** Three papers report it numerically:
**2607.09739** (0.050 vs 0.084 MRE), **2607.05891** (up to +10.02 pp at k=16), **2606.13732** (FID 71 vs
106). One reports random **winning**: **2606.21916** at high pruning rates. One **omits it entirely and the
omission matters**: **2607.17043**. The honest summary of the measured gain is **modest and
budget-dependent** — 2606.21916's ~2 pp over random at moderate pruning is the most sober number in the
scan, and it comes from the only paper whose purpose is to audit rather than propose.

**On the encoder-bias question (brief item 5).** Ranked by how squarely they face it:
1. **2607.09739** — faces it and is the only paper that makes model-independence of the selection
   signal an explicit design goal. Still leaves the selecting-encoder's own bias untested.
2. **2606.13732** — proves the mechanism (biased reference prunes tail modes, power-law diversity decay)
   without framing it as an encoder question.
3. **2606.01304** — the retrieval-side statement: negatives "selected by retriever score rather than
   diagnostic value," contamination **increasing** as the retriever improves.
4. **2607.17043** — discloses the verifier as a distillation channel in Limitations.
5. **2607.19848** — treats encoder-choice flexibility as an unqualified virtue; **runs no sensitivity
   analysis.** Clearest un-priced free parameter in the scan.
6. **2607.05891** — selects in teacher space, evaluates a student, never raises the question.
7. **2606.21631, 2606.03001, 2606.24998, 2607.02637, 2607.09100, 2607.27660** — silent.

**Net:** the field has the mechanism (2606.13732) and one paper with the right instinct (2607.09739), but
**no paper in the 2606–2608 window runs the direct experiment: curate a corpus with encoder A, evaluate
encoder A vs a held-out encoder B, and report the inflation.** That experiment is unclaimed.

**On laptop feasibility (consumer constraint).** Streaming/incremental: **2606.03001** (HNSW, never needs
the full corpus). Embed-once + cheap objective: **2607.09739** (sparsified FL, O(αn²/k)), **2607.05891**
(Euclidean only), **2607.17043**'s KBU (kernel entropy). O(n²)-bounded but fine at 10^3: **2607.19848**
(explicitly OOMs at 100k, irrelevant at 1k). Expensive: **2607.24332** (O(n²) cross-encoder forwards),
**2606.24998** (pretraining scale).

---

## 16. What was searched for and NOT found

Searches run (WebSearch, then filtered by arXiv YYMM prefix 2606/2607/2608):
embedding deduplication synthetic data curation near-duplicate; coreset selection random baseline matched
budget; semantic dedup removing hard examples across the class boundary; cluster-then-sample / mode
collapse / clustering for curation; LLM data quality filters validated on downstream tasks rather than
model opinion; data selection encoder bias / same-encoder evaluation confound; facility location and
submodular diversity selection for instruction tuning; hard-negative mining and false negatives under
similarity thresholds; influence-function and gradient-based selection; dedup threshold selection and
pruning ablations; embedding diversity for safety-guardrail classifier data; pruning that removes
informative boundary examples; and a general 2608 sweep.

**Not found in the 2606–2608 window:**

1. **No paper measuring what semantic dedup deletes.** Nothing that takes a labelled corpus, applies a
   cosine/embedding dedup at threshold τ, and reports the label composition or difficulty distribution of
   what was removed. This is the brief's central question and it is unaddressed. **2607.24332** is the
   nearest miss (it argues the distinction matters) but reports only a 9.75% removal rate, not what the
   9.75% consisted of.

2. **No threshold-selection methodology.** No paper derives a principled dedup threshold. CuratorKIT ships
   0.92 as a configured default with no ablation; no 2606+ paper sweeps τ and reports the downstream curve.

3. **No direct encoder-bias experiment.** Nothing curates with encoder A and evaluates A vs a held-out B.
   **2607.19848** has the tooling to answer it and explicitly does not. **2606.13732** proves the general
   mechanism in a generative-diffusion setting, not an encoder setting. *This is the clearest open gap and
   it sits exactly where this project's guard work lives.*

4. **No quality filter validated against a downstream task in-window.** Every relevant "quality filtering
   validated downstream vs another model's opinion" hit was out of scope (2510.00866, 2412.02980,
   2409.16341, 2406.12397 and similar). Within 2606–2608 the closest is **2607.17043**, which validates
   downstream but has no random arm and uses an LLM verifier — i.e. still partly another model's opinion.

5. **No cluster-purity-as-quality-signal paper.** Cluster-then-sample appears only as *criticized prior
   work* (in 2607.02637, for reducing downstream diversity). No in-window paper proposes cluster purity as
   a validated quality signal.

6. **Nothing on bi-encoder policy-guard data curation specifically**, and nothing on zero-shot
   **policy** transfer at ~1000-label scale. All in-window coreset/selection work is vision classification
   (2607.05891, 2607.09100, 2607.02637), LLM benchmark subsetting (2607.09739), or instruction tuning
   (2607.17043). The guard-specific hits surfaced by search were all 2605 or earlier and are excluded.

7. **Almost nothing dated 2608.** Today is 2026-08-02; the August listing is one day old and no 2608 paper
   surfaced in any of the searches. This scan is effectively 2606–2607 and should be re-run in ~3 weeks to
   pick up August.

8. **No in-window influence/gradient-based selection paper.** All influence-function and gradient-selection
   hits (2602.17835, 2605.21422, 2601.13697, 2510.26491) predate the cutoff and are excluded. The
   difficulty/influence axis of the brief is therefore represented in-window **only** by the negative
   result **2606.21916** — which is itself the most useful thing found on that axis.
