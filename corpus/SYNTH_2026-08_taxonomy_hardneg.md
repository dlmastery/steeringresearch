# SYNTH 2026-08 — Taxonomy-Driven Synthesis of Hard Negatives / Hard Positives

**Scan date:** 2026-08-02
**Hard date filter:** arXiv IDs `2606`, `2607`, `2608` only (i.e. announced 2026-06 or later). Anything `2605` or
earlier is OUT OF SCOPE for citation in this file and is named only as lineage.
**Verification:** every arXiv id below was WebFetched and its title + authors confirmed against the arXiv abstract
page (or the `arxiv.org/html/` full text where noted). No id here is cited from memory.
**Consumer:** the activation-steering program — contrastive (harmful, harmless) prompt pairs for steering
directions, and the `biencoder_guard` policy-matching tower. Filters applied throughout: *does it need a frontier
model in the loop?*, *does it scale to ~1000 labels?*, *does it control for trivial separability?*

---

## Summary table

| Paper | arXiv | Date | Method in one line | Needs frontier model? | Scales to ~1000 labels? | Confound-controlled? |
|---|---|---|---|---|---|---|
| HaloGuard 1.0 | 2607.02079 | 2026-07-02 | 46-policy / 490-category / 2,940-subcategory constitution drives exhaustive 1:1 paired counterfactuals (topic + vocabulary held fixed, intent flipped) | **Yes** — Qwen 397B, GPT-OSS-120B, Grok 4.20, Qwen Max routed per language | **Yes, demonstrated at 2,940 leaves** | **Structurally, not measurably.** Topic/vocab pinning + length-band targeting with drift pre-compensation; **no reported length/lexical/n-gram separability number** |
| SingGuard-NSFA | 2607.13081 | 2026-07-13 | 7→28→185 CIA-triad risk taxonomy, cross-checked against 3 OWASP lists; 4-stage seed-free→seed-based→multilingual→verify pipeline; 3-source adaptive negative sampling | **No** — ensemble of 74 *open-source* LLMs as generators; Qwen3.5 annotator/verifier | **Yes, 185 leaves × ~1,000 samples each** | **Partly.** Template-level leakage prevention (disjoint train/eval prompt templates), MinHashLSH dedup across the train/eval boundary, 7-model majority-vote labels. **No lexical-overlap or length analysis** |
| CausalNeg ("When Hard Negatives Hurt") | 2606.01304 | v1 2026-05-31 / v2 2026-06-07 | CoT decomposes *why* a positive satisfies the query into K requirements, then surgically violates one; + query-view entropy maximization to suppress source shortcuts | Generator is "a strong proprietary instruction-tuned LLM" (unnamed); trained model is Qwen3-0.6B | Retrieval-scale, not label-taxonomy-scale | **Yes — the only paper here that makes the shortcut the object of study.** Formalizes I(Z; s(q,d⁻)) > 0 as the failure. Diagnostics are gradient-space, **not** surface-space |
| Synthetic minority data is redundant or invalid | 2607.20787 | 2026-07-22 | Redefines synthetic-data validity as a population quantity estimated against *withheld real* data; proves an invalidity floor set by class overlap | No | N/A (it is a test, not a generator) | **Yes, and it is the sharpest instrument in this scan** — it exists to de-bias a validity check that "cannot fail" |
| Vera / Safety Testing LLM Agents at Scale | 2607.01793 | 2026-07-02 (v2 07-04) | Literature-driven taxonomy discovery → combinatorial composition of risk × attack × environment → sandboxed execution with deterministic verification predicates | Not stated; targets are frontier agents | **Yes** — 124 leaf risk × 77 leaf attack × 30 leaf environment | Validity is **execution-grounded** (environment state + tool calls), which sidesteps text confounds entirely |
| DT-Guard | 2607.06326 | 2026-07-07 | Intent→Category→Safety distillation over 1.9M aggregated real samples; **Borderline class defined by annotator disagreement**; RG-PHO 3-rollout hard-case stratification | GLM-5.1 as distillation annotator | 9 risk categories only — **no** | **No** — no length, lexical-overlap or template analysis reported |
| Principles of Concept Representation in Sentence Encoders | 2606.06994 | 2026-06-05 | Controlled ablation of encoder conditions on 3.3M WordNet/Wiktionary synonym + definition pairs (= hard positives at scale), evaluated on 3 decontaminated splits | No | Concept-family scale, yes | Decontaminated splits; **no surface-confound measurement reported** |
| Real-Time Hard Negative Sampling via LLM-based Clustering | 2607.00448 | 2026-07-01 (v2 07-06) | LLM learns item representations → cluster → draw negatives from *within the positive's cluster*, in real time during training | LLM used for representation/clustering; size not disclosed | Billions of training points; cluster count is a free parameter | **No** — but it does report breaking feedback loops and reducing popularity bias |
| OntoExtend | 2607.17963 | 2026-07-20 | Requirement-driven (competency-question) RAG extension of an *existing* ontology rather than generation from scratch | Not disclosed | Evaluated at 39 CQs — small | N/A (taxonomy, not example synthesis) |

---

## Per-paper detail

### 1. HaloGuard 1.0: An Open Weights Constitutional Classifier for Multilingual AI Safety — arXiv:2607.02079
Navaneeth Sangameswaran, Preetham S, Ashmiya Lenin. Submitted 2026-07-02. *(Full text read at `arxiv.org/html/2607.02079`.)*

**Method.** The taxonomy *is* the corpus schema, not a labelling convention applied afterwards. 46 constitutional
policies (29 harmful + 17 shared-harmless) → 490 categories (380 + 110) → 2,940 subcategories (2,416 + 524). Each
harmful subcategory carries two pieces of confusion metadata that are the actually novel part: `surface_cues`
(tokens it shares with legitimate uses — for arson: "accelerant", "pour pattern", "ignition point") and
`benign_confusions` (the specific legitimate adjacent uses — "fire-investigation analysis", "fire science
education"). Hard negatives are then **not** sampled; they are *generated from the confusion metadata*, exhaustively
1:1: "155,769 benign twins", "231,231 records carry a `paired_with_id`". The benign twin holds topic and surface
form fixed and flips intent through one of nine legitimising angles (legal, historical, scientific, journalistic,
clinical, …). Generation is style-anchored — real prompts are retrieved and used to condition surface register, so
the synthetic side does not read as "clean". Two-tier harmless design separates *boundary* false positives
(constitution-local, shares vocabulary with the harmful policy) from *baseline* false positives (ordinary benign
traffic). Adversarial overlays: 60+ allow-listed attack patterns (74,419 records) and deterministic transforms —
base64/hex/URL/ROT, JSON/XML/markdown wrappers, homoglyph and Unicode substitution, casing shifts (210,396
records). Six quality gates: label consistency, an operational-boundary gate that rejects/rewrites benign records
drifting into operational uplift, refusal backstops, minimum-content filters, a rewrite-mode judge, and rationale
distillation (a teacher that cannot justify the gold label flags the row).

**Numbers.** 1,259,451 records total (train 1,227,290 / eval 21,710 / test 10,451). By bucket: harmful 440,545
(35.0%), harmless-boundary 602,769 (47.9%), shared-harmless 216,137 (17.2%). HaloGuard-0.8B: best average F1 **90.9**
across seven prompt-safety benchmarks, FPR 4.3, FNR 9.5, beating open guards up to 27B. HaloGuard-4B: F1 **92.1**,
FPR 3.5. Taxonomy build is gated: "a surgical YAML writer enforces exact 1:1 coverage, and a missing mirror is a
build error"; a search-backed verification pass over anchors corrected a ~3% citation-error rate during
development. Split hygiene is component-based — a paired counterfactual cannot be split across train and eval.

**Confound status — read this carefully.** HaloGuard controls separability *by construction* (topic + vocabulary
pinned across the pair) and shows awareness of length ("length-band targeting" with "drift pre-compensation: LLM
output tends to drift roughly one band longer than requested"). It **does not report a single post-hoc separability
measurement** — no length AUC, no lexical-overlap statistic, no n-gram artifact probe, no template-leak number. The
register-preservation check emits a `REGISTER:` label but no success rate is reported. So the design is the strongest
in the scan and the *evidence* that the design worked is absent.

**What this project could use:** the `surface_cues` / `benign_confusions` per-leaf metadata schema — it converts
"generate a hard negative" into a *lookup*, which is exactly what makes it runnable on a 1–4B local model.

---

### 2. SingGuard-NSFA: Extensible Guardrails for Agentic AI via Generative Reasoning and Real-Time Classification — arXiv:2607.13081
SingGuard Team. Submitted 2026-07-13. *(Full text read at `arxiv.org/html/2607.13081v1`.)*

**Method.** NSFA taxonomy: 7 Level-1 domains → 28 Level-2 risks → 185 Level-3 variants, grounded in the CIA triad
and built from a systematic survey of industry frameworks. **Coverage is validated externally and honestly**: it
covers 8/10 of OWASP Top 10 for LLM Applications (80%), 8/17 of OWASP Agentic AI Threats and Mitigations (47%), and
6/10 of OWASP Top 10 for Agentic Applications (60%), with uncovered items stated as out of single-turn detection
scope. This is the only paper in the scan that reports a *completeness fraction against an external reference*
rather than asserting coverage. Four-stage pipeline: (1) seed-free generation — an ensemble emits paired risky/benign
samples per risk variant with no seeds; (2) seed-based augmentation — Stage-1 positives become in-context examples,
curated to ~1,000 samples per variant; (3) multilingual expansion to Chinese + 11 high-resource + 120 low-resource
languages; (4) final verification by independent re-annotation, discarding disagreements. Negatives come from an
**adaptive negative sampling strategy with three sources**: paired-generation hard negatives (capped at |P|),
cross-domain negatives (positives borrowed from *other* risk types — the sibling/near-miss mechanic), and safe
No_Risk negatives, bounded at 1:3 positive:negative.

**Numbers.** Training: 935,405 multilingual query samples over 160 risk variants; 274,034 response samples over 25
variants. Benchmarks: query 63,431 (29,474 pos : 33,957 neg), response 29,972 (14,314 : 15,658), CrossSource-Query
3,435 (2,315 : 1,120). SingGuard-NSFA 9B generative: **96.95% F1** query, **98.31%** response, vs best competitor
84.76% / 91.25%; CrossSource-Query **91.29%** vs 84.67%. Discriminative head runs at 45–57 ms/sample vs 3.6–7.2 s for
the generative arm. Models at 0.8B / 2B / 4B / 9B.

**Confound status.** Three real mechanisms, all structural: (i) distinct prompting templates between training and
evaluation, i.e. explicit **template-level leakage prevention**; (ii) MinHashLSH dedup across the train/eval
boundary; (iii) seven-model majority vote (4-of-7) for benchmark labels. Plus internal-consistency filtering
(generator label vs annotator label) and degenerate cleaning (empty, too-short, echo-back). **No lexical-overlap or
length analysis is reported.**

**What this project could use:** **generators are 74 open-source LLMs — no frontier API anywhere in the pipeline.**
This is the single most transferable fact in the scan for a 16 GB host. Also the cross-domain negative source: for a
harmful/harmless steering pair set, positives from a *sibling* harm category are free hard negatives.

---

### 3. When Hard Negatives Hurt: Bridging the Generative-Discriminative Gap in Hard Negative Synthesis for Retrieval (CausalNeg) — arXiv:2606.01304
Zhicheng Zhang, Jiwei Tang, Kuicai Dong, Xiaopeng Li, Jieming Zhu, Jingyu Li, Qianhui Zhu, Fengyuan Lu, Wang
Jiaheng, Gang Wang, Hai-Tao Zheng, Zhaocheng Du. **v1 submitted 2026-05-31, v2 2026-06-07.** *(Full text read at
`arxiv.org/html/2606.01304v2`.)*

> **Date caveat, stated plainly:** the arXiv id prefix is `2606` and so passes the filter as specified, but v1 was
> submitted 2026-05-31 — two days before the 2026-06-02 cut. The v2 revision is 2026-06-07. Treat as borderline.
> This is the paper already cited in the project's CLAUDE.md as "CausalNeg 2606.01304".

**Method.** Two modules. (1) *CoT-guided counterfactual perturbation.* Stage 1 — prompt the LLM to produce a CoT
that decomposes *why* d⁺ satisfies q into K information requirements. Stage 2 — pick one of three perturbation
strategies: **entity substitution** (swap the target entity for a related but incorrect one), **requirement drift**
(shift focus to an adjacent but distinct information need), **constraint violation** (break a temporal, spatial,
causal or quantitative boundary condition). Stage 3 — style-controlled generation, where style references
"constrain the surface form to match the corpus distribution". (2) *Query-view entropy maximization* during
training, dispersing generated negatives across the similarity spectrum to minimise mutual information between
source identity and similarity score.

**Numbers.** nDCG@10, average over mMARCO-zh / HotpotQA / NQ / TriviaQA: mined-only **64.65**, vanilla generation
**1.71** (i.e. total collapse — 0.00 on mMARCO-zh and NQ), naïve mixture (SyNeg) **63.05** (*worse than mined-only*),
CausalNeg **66.55**. Shortcut diagnostics: generated negatives' softmax gradient contribution ratio falls 1.34 → 0.08
across training; cosine similarity between positive and generated-negative gradients weakens 0.73 → 0.24. Encoder
Qwen3-Embedding-0.6B, fine-tuned model Qwen3-0.6B; the CoT generator is "a strong proprietary instruction-tuned LLM"
— **unnamed, so the frontier-dependency of the data step is unverifiable**.

**Confound status — the important one.** This is the only paper in the scan that treats trivial separability as the
central phenomenon rather than a hygiene footnote. It names the failure **source-dependent shortcuts**: "distributional
artifacts enable the model to distinguish negatives by origin rather than relevance", formalised as shortcut learning
occurring when I(Z; s(q,d⁻)) > 0 for Z ∈ {mined, gen}. **But its evidence is gradient-space, not surface-space** — it
reports no AUC of a source classifier, no length statistic, no lexical-overlap number. It proves the shortcut exists
by its downstream damage, not by measuring the surface feature that carries it. For this project that is a gap, not
a disqualification: `confound_report`'s length_auc / count_auc is precisely the missing surface-side instrument.

**What this project could use:** the naïve-mixture result (63.05 < 64.65) is the falsifiable warning — adding
LLM-generated negatives to a mined set *underperformed doing nothing*. Any hard-negative augmentation of the
steering pair set must be tested against a no-augmentation arm, not assumed to help.

---

### 4. Synthetic minority data is redundant or invalid: a data-dependent validity theory and a de-biased test — arXiv:2607.20787
Ahmad B. Hassanat, Ahmad S. Tarawneh, Ghada A. Altarawneh. Submitted 2026-07-22.

**Method and claim.** For two decades the standard evidence that synthetic minority examples are valid has been "a
check that cannot fail: synthetic points are scored against the very data that generated them." The paper de-biases
it: validity becomes a *population* quantity — the probability a synthetic point truly belongs to the minority class
— with a consistent estimator that scores synthetic points **against withheld real data**. It further proves validity
is a property of the data, not the method: class overlap sets an invalidity floor no faithful generator escapes,
making oversampling **redundant where classes separate and invalid where they overlap**.

**Numbers.** Where held-out ground truth is available, the classical test underestimates true invalidity in **96–99%**
of method-by-imbalance-ratio cells, while the de-biased estimator tracks it closely. Across **91 methods**, three
classifiers, and datasets spanning medicine and finance — including a generator engineered to pass the classical
check — none clears both bars: gains over the best trivial baseline are "noise-thin (median below 0.01 F1, a decision
threshold's reach)", and most damage calibration.

**Scope limit, stated:** this is the tabular/SMOTE-family oversampling literature, not text. The *test* transfers; the
91-method survey does not.

**What this project could use:** this is a ready-made falsifier for any synthesis experiment run here. Score generated
negatives against a **withheld real** harmful/harmless slice, not against the pool they were generated from. And note
the structural prediction: the steering setting has *high* class overlap at the boundary — exactly the regime where
the paper predicts synthesis is invalid rather than merely useless.

---

### 5. Safety Testing LLM Agents at Scale: From Risk Discovery to Evidence-Grounded Verification (Vera) — arXiv:2607.01793
Yunhao Feng, Ruixiao Lin, Ming Wen, Qinqin He, Yanming Guo, Yifan Ding, Yutao Wu, Jialuo Chen, Zhuoer Xu, Xiaohu Du,
Jianan Ma, Zixing Chen, Xingjun Ma, Yunhao Chen, Xinhao Deng. Submitted 2026-07-02 (v2 2026-07-04).

**Method.** Three stages: (1) literature-driven taxonomy development that *discovers and structures* risks, attack
methods and environments rather than taking an expert-fixed list — the complete taxonomy holds **124 leaf risk
categories, 77 leaf attack methods, and 30 leaf environment categories**; (2) combinatorial composition, generating
concrete safety cases with **deterministic verification predicates** attached; (3) adaptive execution in isolated
sandboxes with evidence-grounded verification judged from environment state and tool calls rather than model
self-reports.

**Numbers.** Vera-Bench: **1,600 executable safety cases** across 124 risk categories and three execution
environments. Evaluated against four production agent frameworks (OpenClaw, Hermes, Codex, Claude Code), average
attack success rate **93.9%** under multi-channel attacks.

**What this project could use:** the *combinatorial composition* step — risk × attack × environment as an explicit
product grid — is the cheapest known way to turn a small taxonomy into a large, evenly-covered example set. And the
deterministic verification predicate attached at generation time is the validity check this project keeps not
having: the generator emits the test alongside the example.

---

### 6. DT-Guard: Intent-Driven Reasoning-Active Training for Reasoning-Free LLM Safety Guardrail — arXiv:2607.06326
He Liu, Changtao Miao, Xinjie Yang, Tianle Song, Yin Wu, Junchi Chen, Bintao He, Xinyuan Zhang, Bo Zhang, Shi Yan,
Wei Lu, Wei Wang, Danyang Xu, Jiansheng Cai, Zhe Li. Submitted 2026-07-07. *(Full text read at
`arxiv.org/html/2607.06326v1`.)*

**Method.** Not taxonomy-driven *generation* — taxonomy-driven *distillation over real data*, which is a different
and cheaper trade. 1,918,565 raw samples aggregated from six safety domains (red/blue teaming, jailbreak attacks,
alignment data, toxicity, bias, domain-specific risks). GLM-5.1 annotates every sample with a fixed schema: CoT,
intent, risk category, safety label. Intent taxonomy is three-way: Normal 58.08%, Risky (unsafe without attack
technique) 40.56%, Attack (unsafe *with* jailbreak/adversarial method) 1.35%; nine risk categories. **The Borderline
class is defined operationally by annotator disagreement** — multi-round voting keeps unanimous agreement, marks 2:1
splits as Borderline, rejects mismatches. RG-PHO then stratifies by rollout consistency: K=3 independent rollouts per
sample; 3/3 correct = stably mastered, 0/3 = persistently failed, 1/3 or 2/3 = preference-unstable, each routed to a
different optimisation treatment.

**Numbers.** Final dataset 811,897 samples (42.32% retention from the 1.9M corpus): Safe 55.43% (450,000), Unsafe
39.06% (317,134), Borderline 5.51% (44,763) — the stated ≈5.5:4:0.5 ratio. Average F1 **0.886** prompt-side, **0.870**
response-side; dual-side average **0.878** with a 4B backbone, beating 8B guardrail baselines.

**Confound status: none reported.** No length, lexical-overlap or template-artifact analysis appears in the paper.

**What this project could use:** two mechanics, both cheap and both runnable locally. (a) *Borderline = disagreement*
— a hard example is definable without a taxonomy at all, just from judge non-unanimity, which this project can
compute from existing multi-seed judge runs. (b) *K-rollout consistency stratification* — 3 rollouts is enough to
split "mastered / failed / unstable", and the unstable slice is the hard-example mine.

---

### 7. Principles of Concept Representation in Sentence Encoders — arXiv:2606.06994
Isabelle Mohr, John Dujany, Jonathan Souquet, Andre Freitas. Submitted 2026-06-05.

**Method.** Controlled ablation over encoder conditions trained on **3.3 million synonym and definition pairs from
WordNet and Wiktionary** — i.e. hard *positives* obtained from a lexical resource rather than generated — evaluated
on three decontaminated splits plus a modifier-labeled noun-phrase benchmark. Framing is representational
compositionality: an encoder supports a concept family only when its latent space admits a low-distortion
realization of the corresponding semantic operator.

**Four principles as stated.** P1 fine-tuning recalibrates the latent geometry rather than expanding it. P2 semantic
signal concentrates in the final transformer layer before concept-specific training begins, making cross-layer
pooling redundant. **P3 hard negatives improve discrimination and stress-test robustness *without* improving
retrieval ranking — calibration and ranking are independently addressable.** P4 extensional training helps
intersective and subsective concept families while *degrading* relational and intensional ones. Two evaluation
datasets released: a DBpedia semantic-gap benchmark and a modifier-labeled NP paraphrase suite.

**Confound status.** Decontaminated splits are reported; no surface-confound measurement is.

**What this project could use:** P3 is directly aimed at `biencoder_guard`. If hard negatives move discrimination but
not ranking, then the guard's headline metric determines whether hard-negative synthesis is worth any compute at all
— and the two must be reported separately. P2 is a free prior for the probe-layer question. Note P1/P2 were measured
on sentence encoders, not on a decoder's residual stream, so they are hypotheses here, not results.

---

### 8. Real-Time Hard Negative Sampling via LLM-based Clustering for Large-Scale Two-Tower Retrieval — arXiv:2607.00448
Ivan Ji, Liuyi Hu, Harrison (Zihao) Zhao, Lei Huang, Qunshu Zhang, Max (Xiangjun) Fan, Aameek Singh. Submitted
2026-07-01 (v2 2026-07-06).

**Method.** Industry two-tower retrieval standard is in-batch / out-of-batch negatives, which "often produce easy
negatives that models can quickly learn". Instead, an LLM learns media representations, items are clustered, and
hard negatives are drawn **from the same cluster as the positive**, generated in real time during training. Designed
for production integration at billions of training points with minimal computational complexity.

**Numbers.** The abstract reports outperforming widely-used industry methods on public datasets and in an online
deployment, and — the more interesting claim — that the sampling method "can help break inherent feedback loops in
recommendations and significantly reduce popularity bias." No metric values are given on the abstract page; I did
not read the full text, so **no numeric result is asserted here**.

**What this project could use:** the mechanic, not the domain. "Draw the negative from the positive's own cluster"
is the single cheapest hard-negative rule available — it needs one embedding pass and a k-means, no generation at
all, and it is trivially applicable to the existing harmful/harmless prompt pools. It is also the natural control
arm against which any generative hard-negative method should be priced.

---

### 9. OntoExtend: A Framework for Requirement-driven and Scalable Ontology Extension with LLMs — arXiv:2607.17963
Anna Sofia Lippolis, Mohammad Javad Saeedizade, Stefan Schmid, Simon Blattner, Robin Keskisärkkä, Aldo Gangemi, Eva
Blomqvist, Andrea Giovanni Nuzzolese. Submitted 2026-07-20.

**Method.** Extends an *existing* taxonomy rather than generating one from scratch, on the explicit observation that
"current approaches rarely tie ontology extension explicitly to requirements or reusable core models, and offer
limited, systematic evaluation of LLM outputs". Requirements are formalised as **competency questions**; RAG over
relevant input ontologies plus the CQs proposes grounded extensions.

**Numbers.** Evaluated on **39 CQs** across two use cases (the EU-project ontology Onto-DESIDE, and an industrial
Bosch ontology). Generated fragments show few structural issues, satisfy all functional evaluation tests, and are
rated by ontology engineers as requiring **minor to moderate revision** before integration. Performance is sensitive
to CQ specificity and modelling profile. Model names are not given on the abstract page.

**What this project could use:** the competency-question device is the missing coverage instrument. "Does this
taxonomy answer the questions we need answered?" is checkable; "is this taxonomy complete?" is not. At 39 CQs the
evaluation is small and the result is a drafting-assistant claim, not an automation claim.

---

## What I searched for and did NOT find — gaps are findings

Each line below is a search that was run against the 2606/2607/2608 window and returned nothing in-window that
satisfied it. These are the honest holes.

1. **No paper in the window measures trivial separability of its own generated negatives.** Not one of the nine
   reports a length AUC, a bag-of-words / n-gram baseline, a source-classifier AUC on surface features, or a
   token-count distribution for generated vs real negatives. HaloGuard controls length by construction and never
   verifies it; CausalNeg proves the shortcut exists via *gradient* dynamics and never localises it to a surface
   feature; SingGuard prevents template leakage procedurally and never measures residual leakage. **The
   `confound_report` (length_auc / count_auc) discipline this project already runs is, as far as this scan reaches,
   ahead of the published hard-negative synthesis literature.** That is a claim about a two-month window and nine
   papers, not about the field.

2. **No 2606+ paper on taxonomy-driven contrastive-pair synthesis for *activation steering* specifically.** Searches
   for steering-vector / CAA / concept-direction dataset construction in the window surfaced only pre-window work
   (2602.02712, 2605.28664 and earlier) — out of scope by the date filter and not cited here. The guardrail-classifier
   literature is where taxonomy-driven synthesis is actually advancing; steering has not adopted it.

3. **No hard-*positive* synthesis method in the window.** 2606.06994 uses WordNet/Wiktionary synonym+definition pairs
   — a *lexical resource*, not synthesis — and the multi-prototype work that surfaced (2602.10143) is out of window.
   Searches for paraphrase-invariant positive augmentation, multi-prototype concept representation, and
   semantics-preserving surface variation returned no in-window match. Hard positives are the underserved half.

4. **No coverage *guarantee* for automatically-generated taxonomies.** SingGuard reports coverage as a fraction
   against three external OWASP lists (80% / 47% / 60%) and OntoExtend checks against competency questions —
   both are *measurements against a reference*, not guarantees, and both require the reference to exist. Nothing in
   the window offers completeness under a stated assumption. Taxonomy-generation searches otherwise returned
   pre-window work (SC-Taxo 2605.00620) and tooling (OntoLearner 2607.01977, not reviewed here).

5. **No published measurement of whether a 1B–4B *local* model can generate usable hard negatives.** SingGuard proves
   an ensemble of 74 open-source LLMs suffices — with a 122B annotator and a 397B verifier behind it. HaloGuard's
   generators are 120B–397B class. Nobody in the window reports the quality floor as generator size shrinks, which is
   exactly the number this host needs.

6. **No adversarial-mining-against-the-current-model paper in the window for text safety classifiers.** DT-Guard's
   RG-PHO rollout stratification is the closest available (hard cases identified by the model's own inconsistency)
   and it is mining over a fixed corpus, not generating against the current checkpoint. HaloGuard mentions an
   "always-on adversarial red-teaming protocol" without a reported evaluation of it.

7. **The validity literature and the synthesis literature do not talk to each other.** 2607.20787 shows the standard
   validity check is circular and that class overlap sets an invalidity floor; none of the six synthesis papers in
   this scan applies a held-out-real validity estimator to its own generated corpus. Every one of them validates
   generated data with a judge or a vote drawn from the same generation pipeline family.

---

## Bottom line for this program

The three most implementable, in rough order of cost:

1. **Cluster-drawn negatives (2607.00448 mechanic)** — one embedding pass + k-means over the existing prompt pool,
   no generation, no frontier model. This is the control arm every other method must beat.
2. **Confusion-metadata-driven counterfactual pairs (2607.02079 schema)** — attach `surface_cues` and
   `benign_confusions` to each harm leaf, then generate the benign twin with topic and vocabulary pinned. Reduces
   generation to a constrained rewrite, which a 1–4B local model can plausibly do. Verify with `confound_report`,
   which HaloGuard itself does not do.
3. **Held-out-real validity estimation (2607.20787 test)** — score any generated negative against a withheld real
   slice rather than the pool it came from, and expect the paper's structural prediction to bite: high class overlap
   at the harmful/harmless boundary is the regime where it predicts synthesis is *invalid*, not merely redundant.

The cautionary result to pre-register against: CausalNeg's naïve mixture scored **63.05** vs **64.65** for
mined-only. Adding LLM-generated hard negatives made things worse. A no-augmentation arm is mandatory.
