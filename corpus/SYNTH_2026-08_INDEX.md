# Data synthesis for steering — SOTA scan, Jun–Aug 2026

**Scan date:** 2026-08-02 · **Hard filter:** arXiv `2606` / `2607` / `2608` only.
Older work is named as lineage in each file and never cited as evidence.
**38 papers, every id WebFetch-verified** (title + authors). One `[DATE-AMBIGUOUS]` flag.

| survey | file | papers |
|---|---|---|
| Taxonomy generation + hard negatives/positives | [`SYNTH_2026-08_taxonomy_hardneg.md`](SYNTH_2026-08_taxonomy_hardneg.md) | 9 |
| Persona simulation + long-horizon multi-turn | [`SYNTH_2026-08_persona_multiturn.md`](SYNTH_2026-08_persona_multiturn.md) | 15 |
| Embedding selection + dedup | [`SYNTH_2026-08_selection_dedup.md`](SYNTH_2026-08_selection_dedup.md) | 14 |

---

## The three gaps that matter, because each is an experiment nobody has run

The surveys were commissioned separately and converged on the same shape of hole:
**the field generates data enthusiastically and validates it barely at all.**

**1. Nobody measures whether their generated negatives are trivially separable.**
Not one paper in the window reports a length / lexical / n-gram separability number on its
own synthetic negatives. A hard-negative generator that emits length-separable pairs
manufactures precisely the shortcut it claims to remove. This project's `confound_report`
discipline — which caught `trajguard`'s 0.735 completion-length bar and `meerkat`'s 0.675
length tell — is **ahead of the published work**.

**2. Nobody checks that synthetic attacks resemble real ones.**
No in-window synthesis method verifies its output against real attack data. The one
external-validity method that exists (`2606.20708`, teacher-forced simulator probe) makes a
prediction about our own corpora: divergence is +0.09 for engaged users but **+0.40 for
disengaged** ones, with resistance halved 25.1% → 13.5%. Read across, **simulated attackers
never walk away**, so synthetic corpora under-represent abandoned attacks. Testable on
Attack_600 and CSTM-Bench, CPU-only, judge-free.

**3. Nobody curates with encoder A and evaluates on held-out encoder B.**
`2606.13732` proves the mechanism in diffusion — a biased reference model prunes tail modes
with power-law diversity decay — and nobody has run it on encoders. Filtering with the
encoder under test deletes exactly the examples that encoder finds hard, which rigs the
evaluation. **EXP-H already avoids this** by screening distractors lexically rather than by
cosine, and reports `mean_max_cos_to_real` as an audit rather than a filter. The experiment
is unclaimed.

---

## The null to start from, not the assumption of gain

Two in-window results say synthesis and curation often **do not help**:

- **CausalNeg (`2606.01304`)**: naive generated+mined mixture **63.05** vs mined-only
  **64.65**. Adding generated negatives *hurt*.
- **Data Pruning (`2606.21916`)**: **random selection wins** at high pruning ratios.

Any pipeline built here is pre-registered against those, not against a hoped-for
improvement. And `AdvGRPO` (`2606.09701`) documents that self-play attackers **collapse to
surface paraphrase** — so diversity must be measured structurally, never lexically.

## Cheapest first moves

| move | source | cost |
|---|---|---|
| Cluster-drawn negatives as the control arm every generator must beat | `2607.00448` | embedding pass + k-means, no generation |
| Decision density ρ over Attack_600 / CSTM-Bench | `2606.22164` | one CPU statistic |
| Encoder-sensitivity of diversity across EmbeddingGemma / MiniLM / mpnet | `2607.19848` | tooling exists, analysis does not |
| Facility-location selection with a *third-party* encoder | `2607.09739` + `2607.27660` | O(αn²/k) sparsified |

> Internal QA pass — independent external review pending.
