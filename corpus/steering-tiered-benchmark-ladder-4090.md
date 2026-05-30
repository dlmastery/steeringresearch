# Tiered Benchmark Methodology for Steering Research on a 4090 Laptop
### A 'CIFAR-10 -> CIFAR-100 -> ImageNet' style ladder for LLM steering experiments (small Gemma models)

> Philosophy: never run an expensive benchmark to discover a bug a cheap one would have caught. We climb a LADDER of progressively harder, costlier, more realistic benchmarks. Each rung is a GATE: you only spend compute on the next rung if the current rung shows the method is 'going in the right direction'. This mirrors how vision research uses CIFAR-10 as a fast sanity check before CIFAR-100 before ImageNet.

> Hardware reality: a 4090 LAPTOP (16 GB VRAM, not the 24 GB desktop card). Everything below is sized so the model + hooks + activations + a judge fit. Gemma-2-2B-it in 4-bit ~ 2-3 GB; Gemma-3-1B even less; leave headroom for activation caches and a small judge or use an API judge.

==============================================================

## THE LADDER (5 rungs)

    RUNG  NICKNAME        ANALOGY        COST/run     PURPOSE                       GATE TO PASS
    0     UNIT            'does it run'  seconds      mechanics sanity              vector changes logits at all
    1     SMOKE           CIFAR-10       1-3 min      right DIRECTION?              monotonic effect + no collapse
    2     DEV             CIFAR-100      10-20 min    does it GENERALIZE a bit?     beats baseline on held-out concepts
    3     STANDARD        ImageNet-1k    1-3 hours    real comparative result       Pareto-beats prior method
    4     FULL            ImageNet+comp  half-day+    publication / capstone        full multi-axis, all safety evals

Rule of promotion: a method must CLEAR the gate at rung k before it is allowed to consume rung k+1 compute. A regression at any rung sends it back with a logged failure reason.

==============================================================

## RUNG 0 - UNIT (seconds) - 'does the plumbing work'
Not a benchmark, a mechanics check. Run before any experiment.
  - Inject the steering vector; assert the residual stream actually changed at the target layer (||delta h|| > 0).
  - Assert logits move; assert special tokens (<bos>, <start_of_turn>) are excluded from steering.
  - Assert removing the vector restores baseline exactly (no state leak).
  Data: 5 hand-written prompts. Pass: deterministic, instant.

## RUNG 1 - SMOKE (CIFAR-10 analog, 1-3 min) - 'are we going the right direction?'
The fast directional sanity test. SMALL, FIXED, CHEAP. Run on EVERY iteration of the autoresearch loop.
  Datasets (mini slices):
    - AxBench-mini: 10-20 concepts only.
    - MMLU-tiny: a fixed 200-question subset (capability tripwire).
    - WikiText-ppl-mini: 50 passages (coherence tripwire).
    - JailbreakBench-mini: 20 of the 100 harmful prompts (safety tripwire).
  What it proves (the 'right direction' signals):
    1. MONOTONICITY: behavior score rises with alpha over a small sweep {0, 0.5, 1.0} - the effect exists and is ordered.
    2. NO-COLLAPSE: PPL stays bounded and MMLU-tiny drop < 5% at the working alpha.
    3. NO-SAFETY-LEAK: JailbreakBench-mini compliance stays ~0%.
  Gate to RUNG 2: monotone behavior + bounded PPL + no safety leak. If monotonicity fails -> wrong layer/direction; fix before spending more.
  Cost on 4090 laptop: minutes; this is your inner-loop dial.

## RUNG 2 - DEV (CIFAR-100 analog, 10-20 min) - 'does it generalize a little?'
Harder, more classes, held-out split. Run when a method passes SMOKE and you want to believe it.
  Datasets:
    - AxBench: ~100 concepts INCLUDING a held-out set never used for extraction (the generalization test - like unseen CIFAR-100 classes).
    - MMLU (full or 2-subject) + ARC-Challenge: capability.
    - WikiText-103 perplexity (standard slice): coherence.
    - JailbreakBench (full 100) + XSTest (over-refusal dual): safety + over-refusal.
    - CAA behavior suite (sycophancy/refusal): a second behavior family to check it's not concept-specific.
  Gate to RUNG 3: beats the chosen baseline (prompting OR CAA) on held-out concepts at MATCHED coherence, with safety leak <= baseline and MMLU drop < 2%.

## RUNG 3 - STANDARD (ImageNet-1k analog, 1-3 h) - 'is this a real result?'
The comparative benchmark you would put in a results table.
  Datasets (full):
    - AxBench full + 500-concept holdout (the FLAS/PSR reporting standard).
    - MMLU + ARC + GSM8K + HellaSwag: full capability panel.
    - WikiText-103 + repetition/MAUVE + judge-coherence.
    - JailbreakBench + HarmBench + AdvBench + StrongREJECT: full safety panel.
    - XSTest: over-refusal.
    - Persona/IFEval: trait & instruction control.
  Protocol: LLM-as-judge with calibration (validate precision on a human slice, target ~94% like Rogue Scalpel); >=1000 random vectors for any random-baseline claim; report mean +/- CI; compare at MATCHED coherence not matched alpha.
  Gate to RUNG 4: Pareto-dominates the prior method (safety x capability x behavior) - i.e. no axis regresses.

## RUNG 4 - FULL (publication / capstone, half-day+) - 'defend it'
  Everything in RUNG 3 plus:
    - Cross-model: Gemma-2-2B AND Gemma-3-1B (and 9B if a desktop/cloud burst is available) to show scale-robustness.
    - Red-team: reproduce the Rogue-Scalpel 20-vector UNIVERSAL attack against the guarded method (must neutralize).
    - Ablations: each component on/off (e.g. guard layers A-E; PSR gain on/off).
    - Stress: OOD prompts, jailbreak suffixes, long-generation drift (512+ tokens).
  Gate: full multi-axis win with ablation-justified attribution of every gain.

==============================================================

## 4090-LAPTOP EXECUTION NOTES
  - Model: Gemma-3-1B-it (smoke/dev default, smallest) and Gemma-2-2B-it (standard), both 4-bit (bitsandbytes) or GGUF.
  - Judge: prefer a small local judge only at RUNG 0-2 (or rule-based refusal detector); use a stronger/API judge at RUNG 3-4 to save VRAM.
  - Batch + cache: precompute and cache contrast activations once; reuse across the whole ladder.
  - enforce_eager / hooks: use HF hooks (not vLLM) when activation editing needs per-layer access on 16 GB.
  - Determinism: greedy decoding for safety/efficacy gates; fixed seeds; pin dataset subsets so SMOKE is comparable across iterations.
  - Time budget heuristic: keep RUNG 1 under ~3 min so you can run it dozens of times a day inside the autoresearch loop.

## WHY THIS ORDERING SAVES COMPUTE (the CIFAR lesson)
  - 90% of bad ideas die at RUNG 0-1 in minutes (wrong layer, wrong sign, off-manifold collapse).
  - Only directionally-correct methods reach the expensive safety/HarmBench panels.
  - The SAME axes are measured at every rung (behavior, capability, coherence, safety, selectivity) - only the SIZE and REALISM grow - so a SMOKE pass is a true predictor of a STANDARD pass, exactly like CIFAR-10 accuracy predicting (imperfectly but usefully) ImageNet viability.

## PROMOTION/DEMOTION LOG (what the harness records each rung)
    method_id | rung | behavior | dMMLU | PPL | CR(jailbreak) | over_refusal | verdict(promote/demote) | failure_reason

*Dataset choices are standard community benchmarks; pin versions at run time. Inherited Gemma-specific numbers remain [NEEDS VERIFICATION].*
