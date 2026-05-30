# H20 — Fisher-ratio optimal layer selection (E2)

> Back to master registry: [IDEA_TABLE.md](../../IDEA_TABLE.md)

## TL;DR

The injection layer for a steering vector is typically chosen by hand (a fixed
"late layer" heuristic). This idea tests whether the layer of maximum Fisher
linear discriminant ratio between positive and negative contrast activations
is systematically the best injection layer, as measured by downstream behavior
efficacy. If confirmed, layer selection becomes automatic and behavior-specific,
replacing the current heuristic.

## Citation

Turner, Alexander, Lisa Thiergart, David Udell, Gavin Leech, Ulisse Mini, and
Monte MacDiarmid. 2023. NeurIPS Workshop 'Activation Addition: Steering Language
Models Without Optimization' (arXiv:2309.05907) — introduces CAST and the idea
of selecting layers based on separation quality of the target concept.

Primary CAST paper: Zou, Andy, Long Phan, Sarah Chen, James Campbell, Phillip
Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin
Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex
Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, J. Zico Kolter,
and Dan Hendrycks. 2023. 'Representation Engineering: A Top-Down Approach to AI
Transparency' (arXiv:2310.01405) — per-layer analysis of representation quality
motivates Fisher-based layer selection. The Spearman >= 0.7 threshold is a
pre-registered prediction for this project [NEEDS VERIFICATION].

## Hypothesis

Axis 1 (WHERE / site) is the axis under test. The Fisher linear discriminant
ratio F(l) = (mu_pos - mu_neg)^2 / (var_pos + var_neg) projected onto the mean-
difference direction measures how separable the behavior representation is at
layer l. The hypothesis is that F(l) is a strong predictor of steering efficacy
at layer l: layers where the model has already cleanly encoded the behavior
concept allow a more targeted intervention, while layers where the concept is
still entangled with noise require larger alpha and produce more collateral
damage. The cited CAST / RepEng work shows that linear separability varies
dramatically across layers, with the most separable layer typically appearing in
the upper-middle residual stream. Confirming Spearman >= 0.7 between F(l) and
efficacy(l) across three behaviors would validate Fisher-ratio layer selection
as the default automatic procedure for all downstream experiments, replacing the
fixed-layer heuristic. This directly impacts every subsequent experiment's
extraction quality (Axis 7: HOW DERIVED is coupled to Axis 1: WHERE).

## Falsifier

If the Spearman rank correlation between per-layer Fisher ratio and per-layer
behavior efficacy is < 0.7 across all three tested behaviors (refusal, sentiment,
factuality) on Gemma-2-2B-it (4-bit) at Rung-1 SMOKE, this idea is DISCARDED
and layer selection must use a held-out behavior-score cross-validation sweep
instead of the Fisher heuristic.

## Predicted delta range

All thresholds are [NEEDS VERIFICATION] pre-registered predictions.

| metric | value | rationale |
|--------|-------|-----------|
| Spearman(Fisher, efficacy) | >= 0.7 across 3 behaviors | CAST/RepEng per-layer analysis [NEEDS VERIFICATION] |
| Optimal layer range (refusal, Gemma-2-2B) | layers 14-20 | Rogue Scalpel (2509.22067) fragility peaks early-mid [NEEDS VERIFICATION] |
| MMLU drop at Fisher-optimal vs heuristic layer | < 1 pp | better layer = less collateral [NEEDS VERIFICATION] |
| Behavior efficacy gain over heuristic | +2 to +8 pp | per-layer variation observed in RepEng [NEEDS VERIFICATION] |

## Status

| stage | done? |
|-------|-------|
| implementation written | [ ] |
| tests green | [ ] |
| AUDIT.md filed | [ ] |
| IMPROVEMENTS.md addressed | [ ] |
| VERIFY.md sealed | [ ] |
| First experiment archived | [ ] |
| Verdict authored | [ ] |

## How to test this idea (idea-specific experiment strategy)

1. Cache activations at ALL layers (once) for the contrast pairs (pinned seed,
   AxBench refusal + sentiment + factuality behaviors).
2. Compute Fisher ratio at every layer.
3. Measure behavior efficacy by running SMOKE (Rung 1) at each layer with alpha=0.8.
4. Compute Spearman(Fisher_ratio_vec, efficacy_vec) across layers.
5. Pre-register the predicted range [14, 20] for refusal on Gemma-2-2B BEFORE
   running (not after seeing the plot).
6. Cross-validate: run SMOKE at optimal_layer and at optimal_layer +/- 2;
   confirm the selected layer has highest behavior score.
7. Repeat for all three behaviors; report Spearman per behavior.

The guard-layer C constraint (avoid empirically fragile mid-layer band) from
`skills/steering-rogue-scalpel-guard/SKILL.md` must be applied when selecting
among equally-scoring layers.

## Cross-idea interactions

Composable with:
- E1 (10_diffmean_paircount_knee) — E1 provides the pair count N; E2 uses that
  N to compute Fisher ratios. Both are Block A infrastructure.
- E4 (cosine alignment diagnostic) — uses the optimal layer from E2 as its
  primary layer for the DiffMean vs PCA-top1 comparison.
- ALL downstream ideas — E2's output (optimal layer per behavior) is a shared
  input to every subsequent experiment's extraction config.

Competes with: none (infrastructure; produces a shared resource).

Combination suite: E1 + E2 + E4 = standing extraction infrastructure (Block A).

See also: `skills/steering-vector-extraction/SKILL.md` Section 5 (Fisher ratio
protocol and pre-registration procedure for E2).
