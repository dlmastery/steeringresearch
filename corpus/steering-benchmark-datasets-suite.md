# Comprehensive Benchmark & Dataset Suite for LLM Steering Experiments
### For the 50-experiment autoresearch program on small Gemma models (Gemma-2-2B-it / Gemma-3-1B), RTX 4090

> Purpose: a single, reusable measurement substrate so every experiment (E1-E50, N1-N12) reports against the same axes. Steering has FOUR things to measure simultaneously: (1) did the intended behavior change? (2) did general capability survive? (3) did the text stay coherent? (4) did SAFETY survive (the Rogue Scalpel axis)? A dataset is chosen for each axis, plus extraction/contrast data and an LLM-as-judge protocol.

---

## 0. The five measurement axes (what every experiment logs)

    AXIS                     PRIMARY METRIC                         GOOD = 
    1 Behavior efficacy      behavior success / concept score        high
    2 Capability retention   MMLU / ARC / GSM8K accuracy delta        ~0 drop
    3 Coherence              perplexity, repetition rate, judge-coh   low PPL
    4 Safety integrity       JailbreakBench Compliance Rate           ~0% (no leak)
    5 Selectivity (gated)    harmful-refusal vs harmless-refusal      high gap

---

## 1. CONTRAST / EXTRACTION datasets (to BUILD steering vectors)

These supply the +/- pairs whose difference-of-means (or PCA) gives the steering/condition vectors.

- **AxBench concept set (Stanford, 2501.17148)** - the canonical fine-grained concept steering benchmark; supplies concepts + the eval harness; primary apples-to-apples bench (prompting vs probes vs SAE vs steering). Use its 500-concept holdout for generalization (the FLAS/PSR papers report on it).
- **Anthropic model-written eval / CAA behavior sets (sycophancy, corrigibility, hallucination, refusal)** - the Rimsky/Panickssery CAA contrast pairs; standard for behavior-vector extraction.
- **Sorry-Bench / Alpaca** - harmful vs harmless instruction pairs for refusal/condition-vector extraction (CAST-style).
- **GemmaScope SAE feature dictionaries (google/gemma-scope) + Neuronpedia** - pre-trained interpretable feature directions for Gemma-2-2B/9B; source of SAE-feature steering vectors and the Rogue-Scalpel benign-feature attack set.
- **TruthfulQA contrastive (true vs false statements)** - for ITI/truthfulness direction extraction.
- **Sentiment / Style corpora (IMDB, Yelp, Shakespeare, formality)** - for sentiment/style steering vectors (ActAdd-style 'Love'-'Hate').
- **Persona trait descriptions (evil, sycophancy, hallucination)** - for Persona-Vectors (Anthropic, 2507.21509) automated extraction from natural-language trait descriptions.

## 2. BEHAVIOR-EFFICACY datasets (Axis 1 - did it work?)

- **AxBench** (concept incorporation score) - main metric for 'is the concept present', with coherence control.
- **TruthfulQA (MC + generative)** - truthfulness steering (ITI's home benchmark).
- **CAA behavior suites** - sycophancy / refusal / hallucination behavior rates.
- **Persona steering eval (DailyDilemmas, trait expression)** - for persona/character-trait control (used by AntiPaSTO, Persona Vectors).
- **Sentiment flip / detoxification (RealToxicityPrompts)** - classic ActAdd target.
- **Format/instruction control (IFEval)** - 'follow this constraint' steering (used by PSR/SpotLight).

## 3. CAPABILITY-RETENTION datasets (Axis 2 - did we break the model?)

- **MMLU** - the standard capability-tax probe (Rogue Scalpel uses it to define the alpha 'sweet spot'; <1-2% drop = acceptable).
- **ARC-Challenge / ARC-Easy** - reasoning retention.
- **GSM8K** - arithmetic/multi-step reasoning retention (catches subtle degradation).
- **HellaSwag** - commonsense retention.
- **MT-Bench / AlpacaEval** - open-ended helpfulness retention (used by StTP/StMP coherence studies).

## 4. COHERENCE datasets/metrics (Axis 3 - is the text still fluent?)

- **WikiText-103 perplexity** under steering - the off-the-shelf fluency probe; locate the coherence cliff (E3).
- **Repetition / degeneration rate** (n-gram repeat, MAUVE) on open-ended generation.
- **Judge-coherence score** (LLM-as-judge fluency rating) - AxBench reports steering quality CONTROLLING for coherence; replicate that control.

## 5. SAFETY-INTEGRITY datasets (Axis 4 - the Rogue Scalpel axis)

- **JailbreakBench (100 prompts, 10 categories)** - THE benchmark from Rogue Scalpel (2509.22067); baseline compliance must be 0%; measure Compliance Rate (CR) under every steering condition. Mandatory for any stacking/guard experiment.
- **HarmBench** - broader harmful-behavior eval; cross-check CR generalization.
- **AdvBench (harmful behaviors + strings)** - refusal-rate target (GER-steer/Selective report on it).
- **StrongREJECT** - stricter jailbreak scoring to avoid judge over-counting.
- **XSTest (over-refusal)** - the dual: benign prompts that LOOK harmful; measures the OVER-refusal cost of safety steering (E42).
- **Universal-attack construction set** - reproduce the 20-vector averaging attack (F5) as a red-team probe against any guarded model.

## 6. SELECTIVITY / CONDITIONAL datasets (Axis 5 - did the gate fire correctly?)

- **CAST condition sets (harmful-category vs benign)** - 'if hate-speech then refuse' rule eval; harmful-refusal vs harmless-refusal gap.
- **OR/AND multi-condition mixtures** - synthetic mixtures across N safety categories to test condition-vector orthogonality (E10/E11).
- **Out-of-distribution prompt sets (new domains, paraphrases, jailbreak suffixes)** - gate robustness under shift (E15, E41).

## 7. EVALUATION PROTOCOL (shared judge + scoring)

- **LLM-as-judge:** a strong instruct model in reasoning mode (e.g. Qwen3-8B as in Rogue Scalpel, or a Llama-3.x judge) classifying SAFE/UNSAFE and rating coherence; validate judge precision against a small human-annotated slice (Rogue Scalpel reports 94% precision on the harmful class - replicate this calibration).
- **Rule:** incoherent/repetitive outputs scored SAFE for the harm metric but FAIL coherence - so a method can't 'win' safety by producing gibberish.
- **Decoding:** greedy for safety/efficacy reproducibility; sampled for coherence/diversity probes.
- **Stats:** sample >=1000 vectors for random-baseline experiments (Rogue Scalpel standard); report mean CR + CIs; pre-register thresholds.

## 8. Minimal vs full tiers (for a 4090 budget)

    TIER         USE WHEN                 DATASETS
    Smoke (fast) every iteration loop     AxBench-mini + MMLU(500) + WikiText-ppl + JailbreakBench(100)
    Standard     per-experiment report    + ARC + GSM8K + XSTest + HarmBench + CAA suite
    Full         publication/capstone     + TruthfulQA + AdvBench + StrongREJECT + MT-Bench + persona evals + 500-concept holdout

## 9. Coverage matrix (experiment block -> datasets)

    BLOCK A (tooling)        AxBench, MMLU, WikiText-ppl
    BLOCK B (conditional)    CAST conditions, JailbreakBench, XSTest, OOD sets
    BLOCK C (stacking)       AxBench(multi-concept), JailbreakBench, MMLU, WikiText
    BLOCK D (geometry)       AxBench, WikiText-ppl, MMLU (1B vs 9B)
    BLOCK E (mechanism)      AxBench, GemmaScope features, TruthfulQA, AdvBench
    BLOCK F (safety/robust)  JailbreakBench, HarmBench, StrongREJECT, XSTest, AdvBench, MMLU
    NOVEL N1-N12             all axes; N5/N3 add activation-statistics probes (norm, participation ratio) on WikiText activations

*Datasets named here are standard community benchmarks. Confirm exact splits/versions at run time; treat any inherited Gemma-specific result as [NEEDS VERIFICATION].*
