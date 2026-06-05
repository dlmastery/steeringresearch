# GLOSSARY — every term, in plain English

> **Read this if the hypothesis pages feel too technical.** This defines, in
> normal language, every recurring term used across the project. Each hypothesis
> page also has its own **"In Plain English"** box at the top that defines the
> terms *it* uses — so you never have to be a specialist to understand what a
> page is testing or why.

---

## The big picture (what the whole project is about)

We want to **control how a language model behaves — make it talk about the
ocean, be more formal, refuse harmful requests — without retraining it.** The
trick is to nudge the model's internal "thoughts" mid-sentence. This project
asks, rigorously: *does that nudging actually work, how, and how reliably?* The
honest answer we keep finding: it works a little, and mostly the *strength* of
the nudge matters, not its precise direction.

---

## Core terms

- **Language model (LLM)** — an AI that predicts the next word; here, small open
  models from Google's **Gemma** family (270M and 2B parameters — "M/B" = millions/
  billions of internal numbers; bigger = more capable).
- **Activation / hidden state** — the model's internal "thought vector" at one
  point as it reads/writes text: a long list of numbers (e.g. 2304 of them for
  Gemma-2-2B). We can read it and edit it.
- **Residual stream** — the running internal state that flows through the model's
  layers; the "activation" above lives here. Editing it mid-computation is how we
  steer.
- **Layer** — the model processes text in stacked steps (Gemma-2-2B has 26).
  "Inject at layer 20" = make our edit at step 20. Which step we pick is a knob.
- **Steering** — nudging the model's behavior by **adding a direction to its
  residual stream** while it generates, instead of retraining it. Inference-time,
  cheap, reversible.
- **Steering vector** — the specific direction we add. It points from "not the
  behavior" toward "the behavior" (e.g. from "not-ocean" to "ocean") in the
  model's internal number-space.
- **Contrast pairs** — example sentences that do vs. don't show the behavior
  (an ocean sentence vs. a tax-form sentence). We build the steering vector by
  comparing the model's internal states on these.
- **DiffMean (difference-in-means)** — the simplest way to build a steering
  vector: average the model's internal state on the "yes" examples, average it on
  the "no" examples, and subtract. The difference is the direction. (No training.)
- **PCA-top1** — an alternative way to find the direction: take the dominant axis
  of variation among the per-example differences. Often similar to DiffMean — but
  we found on a real benchmark they're only *moderately* similar.
- **alpha (α) — the steering strength** — how hard we push. Small α = gentle
  nudge; large α = shove. The single most important knob: push too hard and the
  text turns to gibberish (see "coherence cliff").
- **Operation** — *how* we apply the vector: **add** it, **rotate** the state
  toward it (keeping the state's length fixed), or **project it out** (remove a
  direction, e.g. to block refusals). Different geometric ways to nudge.

## How we measure success (the five axes)

- **Behavior efficacy** — did the steered text actually show the target behavior?
  (Our headline measure. Hard to measure honestly — see "judge".)
- **Capability retention** — did the model stay smart (still answer quiz
  questions)? Measured with a multiple-choice test (MMLU).
- **Coherence** — is the text still fluent and sensible, not gibberish or
  repetition? Measured by **perplexity** (how "surprised" the model is by normal
  text — higher = more broken) and/or a fluency rating.
- **Safety** — does steering accidentally make the model comply with harmful
  requests? (Measured on a jailbreak test; any leak is a fail.)
- **Selectivity** — does it wrongly refuse *harmless* requests (over-cautious)?

## How we judge "behavior" (and why it was hard)

- **Judge** — something that reads the steered text and rates whether the behavior
  is present. A weak "proxy" (counting keywords) fooled us early. The valid
  options: an **LLM judge** (another AI reading and rating the text) or a trained
  classifier.
- **Off-family judge** — using a *different* model family to judge (here **Qwen**
  judging **Gemma**), so the judge isn't biased toward its own outputs. We run it
  locally and free, and **validated it against ground truth (ROC-AUC 0.68)** —
  weak but unbiased, and we say so wherever we use it.

## The benchmark we now use

- **AxBench** — a real, published benchmark (Stanford, ICML 2025) of **500
  concepts** with ready-made example text and evaluation prompts. Using it means
  the concepts and tests aren't ones we made up — a key fix, because our earlier
  hand-written single concept ("ocean") *overstated* how well steering works.
- **Shuffled-label control** — a fair comparison vector: same data, but the
  "yes/no" labels randomly scrambled, so it has the same size but no real concept
  meaning. If the *real* steering vector barely beats this, the specific direction
  isn't doing much. (On AxBench, it barely beats it.)

## How we keep ourselves honest (rigor terms)

- **Screening vs. evaluation** — a quick single run (screening) can hint at a
  direction but can't prove it; a real claim needs many repeats + statistics.
- **n** — how many independent repeats. Bigger n = more trustworthy.
- **Wilcoxon / bootstrap CI / Holm** — standard statistics for "is this real or
  luck?": a significance test, a confidence interval (the plausible range of the
  true effect), and a correction for testing many things at once.
- **EXTERNAL-READY** — our strictest bar: a result solid enough to show outsiders
  (passes all the statistics + the worst repeat still beats the best baseline).
  **We currently have zero of these — and we say so.**
- **Verdict words** — SUPPORTED (held up), FALSIFIED (disproven), WEAK/NULL
  (no real effect), PARTIAL, INCONCLUSIVE, UNTESTED (not yet run).

## The headline finding (in one breath)

On a *real* benchmark with an independent judge, the **strength α** (and its
"coherence cliff") is the only knob that strongly matters. *Which* direction,
*which* layer, *which* construction method, *which* operation — all turned out to
matter little. Our earlier synthetic single-concept tests had made steering look
far more precise and powerful than it actually is.
