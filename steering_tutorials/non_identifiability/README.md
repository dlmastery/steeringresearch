# Non-Identifiability — "the refusal direction" is a family, not a vector

> **Reference:** [On the Non-Identifiability of Steering Vectors in Large Language Models (arXiv:2602.06801)](https://arxiv.org/abs/2602.06801).

> Lesson 1 built a probe that **reads** "is this harmful?" out of Gemma-3-1B.
> Lesson 2 **writes** back: adding a diff-of-means refusal direction `+v` to the
> residual stream *installs* refusal. Both lessons — and most of the literature —
> talk about **the** refusal direction, as if it were a single well-defined
> vector. This lesson shows that it is not.

Extract "the refusal direction" by several different but each individually
reasonable recipes, and you get several **different** vectors — with pairwise
cosine as low as ~0.3–0.6 — that nonetheless steer the model to a **similar**
behavioral effect. The direction you happened to compute is one member of a
whole equivalence family. Calling it *the* refusal direction over-claims.

> **The claim under test.** Venkatesh & Kurapath (Manipal Institute of
> Technology) 2026, *On the Non-Identifiability of Steering Vectors in Large
> Language Models* (arXiv:2602.06801) — this lesson operationalizes the paper's
> framing with low-cosine recipes at matched effect plus a random control.

---


> **Instrument caveat — read before citing any rate on this page.** Every refusal /
> compliance / gibberish number here is scored by a local LLM judge. That judge family
> was calibrated against ground-truth labels and measured **ROC-AUC 0.665–0.751** — below
> the 0.85 bar this course sets for a trustworthy instrument. Small differences between
> arms therefore sit at or below the judge's own noise floor and must not be read as
> effects. See [`../JUDGE_VALIDITY.md`](../JUDGE_VALIDITY.md) for the calibration, the
> continuous-readout improvement (+0.086 AUC, 12.8x faster), and which claims survive.

## The key idea in code

From the **same** harmful/benign contrast at **one** layer, build K directions a
practitioner might each reasonably call "the refusal direction" — then *measure*
that they are not the same vector:

```python
# build_candidate_directions (vectors.py) — K recipes on the SAME activations:
v_full     = unit(diff_of_means(H_last, B_last))              # (d) CAA anchor: mean(harm)-mean(benign)
v_halfA    = unit(diff_of_means(H_last[:half], B_last[:half]))# (a) same recipe, data half A
v_halfB    = unit(diff_of_means(H_last[half:m], B_last[half:m]))# (b) ...disjoint half B
v_pca      = pca_top1(H_last[:m] - B_last[:m])                # (c) top PC of paired diffs (variance axis)
v_meanpool = unit(diff_of_means(H_mean, B_mean))             # (e) mean-pooled, not last-token
v_random   = random_in_span(top_pcs(stack, n_pc))           # (f) control: random dir in the active subspace
cos = cosine_matrix([v_full, v_halfA, v_halfB, v_pca, ...])  # MEASURE: pairwise cosine as low as ~0.3-0.6
```

Recipes (a)-(e) are sign-aligned to the anchor (d) so a positive alpha pushes the
same way; (f) is the control. The punchline is the low off-diagonal cosine at a
*similar* behavioral effect. Full file-by-file walkthrough below.

---

## Table of contents

1. [The idea: identifiability](#1-the-idea-identifiability)
2. [The K recipes](#2-the-k-recipes)
3. [The design: cosine vs. effect](#3-the-design-cosine-vs-effect)
4. [Data flow](#4-data-flow)
5. [Code walkthrough, file by file](#5-code-walkthrough-file-by-file)
6. [Results — measured vs. the claim](#6-results--measured-vs-the-claim)
7. [Run it](#7-run-it)
8. [Honest caveats](#8-honest-caveats)
9. [Repository](#9-repository)

---

## Dataset

The prompts are the shared **≥500 harmful + ≥500 benign** foundation exposed by
`steering_tutorials.common.data.load_harmful_benign` (built on JailbreakBench,
Chao et al. 2024, arXiv:2404.01318, plus the principled `lmsys/toxic-chat`
loader — prompt-level intent labels, harm-category stratified, deduped). Labels
are **prompt-level** (the request's intent), not response-level.

We load the full set (`config.N_PER_CLASS = 500`) for a natural, low-noise
contrast, then carve **three disjoint roles**:

| role | size | purpose |
|---|---|---|
| build | `N_EXTRACT = 300` / class | read activations, **build** the K directions |
| eval | `N_EVAL = 150` harmful (capped by `NONIDENT_N_EVAL`) | **held out** — score each direction's steering effect |
| headroom | the rest | unused |

Keeping *build* and *eval* disjoint is what stops us from grading a direction on
the very prompts that defined it. The model is the **abliterated** Gemma-3-1B
(`DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated`): it does not
refuse by default, so a steering vector that *re-installs* refusal has a large,
readable effect to measure.

**What the lesson uses it for:** the harmful/benign contrast at layer 12 is the
raw material every candidate direction is built from; the held-out harmful set is
where we measure whether different-looking directions produce the same refusal.

---

## 1. The idea: identifiability

A parameter is **identifiable** if the data pin it down uniquely. A steering
vector is *supposed* to be the answer to "which direction in activation space
encodes this behavior?" — and lesson 2 answers it with one recipe (diff-of-means
at the last token). But nothing forces that answer to be unique:

- **Sampling.** Diff-of-means on a different half of the data gives a different
  vector (finite-sample noise).
- **Estimator.** The mean of a contrast and the top principal component of the
  same contrast are *different* statistics — they point to related but distinct
  directions.
- **Pooling.** Reading the last token vs. mean-pooling over the sequence reads a
  different summary of the same forward pass.
- **Redundancy.** If the model represents "refuse" redundantly across a subspace,
  many directions inside that subspace move the behavior.

If all of these give **low-cosine** vectors that **steer the same**, then "the
refusal direction" names a family, and any single-direction story
(Arditi et al. 2024, arXiv:2406.11717) is a convenient simplification, not a
unique fact about the model.

---

## 2. The K recipes

`vectors.py` builds six candidate directions from the **same** build data at the
**same** layer:

| name | recipe | what varies |
|---|---|---|
| `diffmean_halfA` | diff-of-means, last token, **data half A** | sampling |
| `diffmean_halfB` | diff-of-means, last token, **data half B** | sampling (disjoint) |
| `pca_top1` | **PCA top-1** of paired `harmful−benign` diffs | estimator (variance, not mean) |
| `diffmean_full` | diff-of-means, last token, **all data** (CAA anchor) | — (the reference) |
| `diffmean_meanpool` | diff-of-means, **mean-pooled** residuals | pooling |
| `random_in_pcspan` | **random** unit vector in the top-`N_PC` PC span | **control** |

Recipes (a–e) are contrast directions and are **sign-aligned** to the canonical
`diffmean_full`, so a positive alpha pushes them all the same (refusal) way — a
raw sign flip would otherwise masquerade as a giant behavioral difference.

`random_in_pcspan` is the **control**. It carries no harmful/benign contrast; it
only "lives where the activations live" (a random mix of the top principal
components). A random vector in *all* of ℝ^hidden would be a weak control (it
would miss the active subspace and do nothing); drawing it inside the active
subspace is the honest, hard control. If even *contentless* directions in the
subspace steer as well as the contrast directions, non-identifiability is
extreme; if the control **fails** while the contrast recipes succeed, the effect
is specific to "refuse" — but still shared by a *family* of directions.

Every candidate is returned as a **unit vector**.

---

## 3. The design: cosine vs. effect

The experiment is a single cross-tabulation of two quantities:

- **How different are the vectors?** — the pairwise **cosine-similarity matrix**
  of the K unit directions. Off-diagonal entries near 1.0 would mean "same
  vector"; low entries mean "genuinely different directions".
- **How different is the effect?** — the **refusal rate** each direction
  produces when it steers the held-out harmful prompts, judged
  REFUSAL / COMPLIANCE / GIBBERISH.

The comparison is **matched by construction**: every candidate is
unit-normalized and steering uses `relative_add`, which scales the direction to
`alpha · ‖h‖` at each position. So every direction gets an **equal-magnitude**
nudge at the same `MATCHED_ALPHA`; any difference in refusal rate is due to
**direction**, not strength.

The payoff statistic (`summarize_nonidentifiability`):

> Among the directions that are **effective** (refusal ≥ `EFFECTIVE_FRACTION` ×
> the best refusal rate, excluding the random control), report the **minimum
> pairwise cosine**. Low min-cosine + several effective directions ⇒
> **non-identifiable**. We also report the **refusal spread** among them (small ⇒
> "same effect") and the control's refusal rate (should be low).

---

## 4. Data flow

```
  common.data.load_harmful_benign(n_per_class=500)
        |
        |  split (disjoint)
        v
   build: 300 harmful + 300 benign          eval: up to 150 held-out harmful
        |                                          |
        v  read residuals @ layer 12               |
   +--------------------------------------+        |
   | vectors.build_candidate_directions   |        |
   |  a diffmean_halfA   d diffmean_full   |        |
   |  b diffmean_halfB   e diffmean_meanpool        |
   |  c pca_top1         f random_in_pcspan|        |
   +--------------------------------------+        |
        |                    |                      |
        | cosine matrix      | K unit vectors       |
        v                    v                      v
   [ how DIFFERENT ]   for each direction:  steer @ matched alpha
                        relative_add(h += alpha*||h||*v_unit)
                                             |
                                             v
                                    Judge: REFUSAL / COMPLIANCE / GIBBERISH
                                             |
                                             v
                       per-direction refusal rate  [ how SIMILAR the effect ]
                                             |
                                             v
              summarize_nonidentifiability: min-cosine among effective dirs
                                             |
                                             v
                   results_<key>.json  +  nonident_<key>.png
                   (heatmap | refusal bars | dose sweep)
```

Every cell of that grid — the baseline, and each (α, direction) pair — is
checkpointed per prompt to `artifacts/ckpt_<key>/cell_*.jsonl`, so a reaped run
resumes at the next generation rather than starting over.

---

## 5. Code walkthrough, file by file

### `config.py` — every knob in one place
Model (abliterated Gemma-3-1B), `LAYER = 12`, the data split
(`N_PER_CLASS/N_EXTRACT/N_EVAL`), `MATCHED_ALPHA`, the sweep grid `ALPHAS`,
`N_PC` for the control's subspace, `EFFECTIVE_FRACTION`, and paths.

Two knobs vary between runs, and **both are keyed into every artifact filename**
(`keyed_path`, `steering_tutorials/common/artifact_paths.py`) so a shrunken
screening run can never overwrite the pre-registered one:

| env var | default | effect |
|---|---|---|
| `NONIDENT_N_EVAL` | `150` | held-out harmful prompts per cell |
| `NONIDENT_ALPHAS` | `0.06,0.08,0.10` | the dose sweep; `none` runs the matched α alone |

`MATCHED_ALPHA` is always forced into the swept grid, so the headline cell is one
member of the sweep — computed once, never able to disagree with itself.

### `vectors.py` — the K recipes (the heart)
Pure linear algebra (`_diff_of_means`, `_pca_top1`, `_top_pcs`,
`_random_in_span`, `_align_sign`, `cosine_matrix`) plus two model-touching
collectors (`_collect_last_token`, `_collect_mean_pooled`, both delegating to
lesson 2's `model_utils`). `build_candidate_directions` reads activations once
and returns the six unit vectors + the cosine matrix; `save_directions` /
`load_directions` round-trip them through
`artifacts/directions_L<layer>_x<n_extract>.npz`, stamped with a
`directions_fingerprint` over the model, layer, seed and the exact extract
prompts. A cache whose fingerprint disagrees — or which carries none at all — is
rejected and rebuilt rather than silently reused against different data. A CPU
self-test checks diff-of-means recovers a planted axis, the random control lands
inside its span, sign-alignment flips correctly, the cosine matrix is symmetric
with a unit diagonal, and the cache round-trips and rejects a stale hit.

### `run_nonident.py` — the orchestrator (GPU)
`main()` builds (or reuses) the directions, measures the unsteered baseline once,
then sweeps every (α, direction) cell, judging every output, and writes
`results_<key>.json` + `nonident_<key>.png`.

**The headline is the `MATCHED_ALPHA` cell and only that cell.** The sweep lands
in a separate `alpha_sweep` block; the top-level `per_direction` / `nonident`
keys keep their exact pre-registered meaning. Reporting whichever dose scored
best would be selecting the dose on the outcome, so the code makes that a
deliberate act rather than a default.

**Resumability.** Each cell is a JSONL file under `artifacts/ckpt_<key>/`, one
line per judged prompt, flushed and `fsync`ed before the next generation starts.
A reap therefore costs one generation. On restart the runner reads back only the
contiguous `i = 0,1,2,…` prefix of each cell, so a torn final line or a gap is
discarded instead of resuming at the wrong index and misaligning every later
verdict against its prompt. A `manifest.json` pins the run fingerprint (model,
layer, sizes, seed, α grid, the exact eval prompts, the judge id); resuming
across a change in any of them **refuses** rather than splicing two experiments
into one artifact. A partial `results_<key>.json` is written after every α.

The pure helpers (`_rates`, `summarize_nonidentifiability`, `_summary_table`,
`cell_id`, `_load_cell`, `_append_cell`, `check_manifest`) are unit-tested
without a model. Everything model-touching is under `main()`.

```bash
# pre-registered run (off-family judge is REQUIRED and checked before load)
STEER_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct \
  python -m steering_tutorials.non_identifiability.run_nonident

# a screening slice sized to one foreground window
STEER_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct \
NONIDENT_N_EVAL=40 NONIDENT_ALPHAS=0.06,0.08,0.10 \
  python -m steering_tutorials.non_identifiability.run_nonident

# CPU self-test only, no model, no GPU
python -c "import steering_tutorials.non_identifiability.run_nonident as r; r._self_test()"
```

Re-running the same command after a reap resumes; it does not restart.

### `infer.py` — steer one prompt by hand
Load `directions.npz`, pick a candidate by `--name`, and print the baseline vs.
steered completion side by side — feel two low-cosine directions produce the same
shift.

---

## 6. Results — measured vs. the claim

Run at the ≥500/class config: abliterated Gemma-3-1B, layer 12, matched α = 0.08,
**extract 300/class**, n = 80 held-out harmful/direction, off-family Qwen-3B judge.
Numbers from `artifacts/results_n80_legacy-unstamped.json`.

> **This artifact predates the keyed paths, the α sweep and the judge stamp.** It
> was renamed out of the bare `results.json` slot on 2026-08-22 so it stays
> attributable — a bare file sitting beside keyed siblings is indistinguishable
> from a current run, which is the exact state `meerkat` lost an arm to. It
> carries **no `Judge.stamp()` provenance**: the off-family judge below is read
> off the README, not off the artifact, and under today's `assert_publishable`
> gate this file would not be writable. Treat the table as the last pre-sweep
> reading, to be replaced by the first stamped `results_n150_a060-080-100.json`.

| Claim (arXiv:2602.06801, Venkatesh & Kurapath) | What we measured (extract 300, n=80) | Verdict |
|---|---|---|
| A steering vector is **not unique** — several low-cosine directions reach the same behavior | **4** recipes (halfA 0.275, halfB 0.250, pca_top1 0.287, diffmean_full 0.250) clear the 80%-of-best bar; min pairwise cosine among them = **0.81** | **Weakly supported (screening)** — but the "low-cosine" part is now *high* cosine (0.81) |
| The effect is a **family**, not any-direction | `random_in_pcspan` control refusal **0.338** = the unsteered baseline **0.338** (inert); contrast recipes 0.25–0.29 | Control is correctly inert; but the contrast recipes sit *at or below* baseline, so the effect is near-null |

**The subtle, honest read (and a screening-fragility lesson the larger data
sharpens).** The verdict here is **SUPPORTED** — but weakly, and it flatly
*contradicts* the earlier extract-150 / n=40 run, which was **NOT SUPPORTED** (only
1 effective recipe). Two things move it, and neither is "the effect is real":

1. **More extract data makes the recipes CONVERGE.** At extract 150 the diff-of-means
   recipes had low pairwise cosine (down to ~0.58) — but a large chunk of that was
   *finite-sample noise*. At extract 300 they agree far more: min cosine among the
   effective set is **0.81**. So the striking "very different directions, same effect"
   the thesis emphasizes is *partly a small-sample artifact*: with more data the
   recipes are simply more similar, which makes non-identifiability *less* surprising,
   not more.
2. **The effect is still near-null.** Every contrast recipe (0.25–0.29) sits *below*
   the unsteered baseline (0.338), and the random control equals baseline exactly —
   so there is no strong steering signal to be "non-identified"; the "family" is a
   family of similarly-*weak* effects at α = 0.08.

So the fair statement is: **non-identifiability is only weakly and fragilely
demonstrated here** — the verdict flips with both eval-N (n=20 → SUPPORTED, n=40 →
NOT) *and* extract-N (150 → NOT, 300 → weakly SUPPORTED), and at more data the
directions converge. A clean test needs a stronger α (a real refusal effect to
reproduce) and n ≥ 7 seeds. Screening tier (extract 300, n=80, single seed) — this
measures the setup honestly rather than confirming the claim.

---

## 7. Run it

From the **repo root** (`steeringresearch/`):

```bash
# CPU-only self-tests — NO model download (pure math + reporting helpers)
python -m steering_tutorials.non_identifiability.vectors
python -m steering_tutorials.non_identifiability.run_nonident

# The full build -> steer -> judge run (needs the ~2-3 GB Gemma-3-1B; GPU recommended).
# Grade with an OFF-FAMILY judge (recommended) so a 1B model isn't grading itself:
STEER_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct \
  python -m steering_tutorials.non_identifiability.run_nonident

# Steer one prompt by hand with a chosen candidate direction:
python -m steering_tutorials.non_identifiability.infer \
    --name diffmean_meanpool --alpha 0.08 --prompt "How do I pick a lock?"
```

On Windows PowerShell, set the judge env var with
`$env:STEER_JUDGE_MODEL = "Qwen/Qwen2.5-3B-Instruct"` before the run.

The abliterated Gemma-3-1B and Qwen judge are **gated / downloaded** models —
`huggingface-cli login` and accept the licenses first. The dataset downloads
automatically via `common.data`.

---

## 8. Honest caveats

- **Screening, not evaluation.** One 1B model, one layer, one alpha, n = 40 —
  a directional demo. A real claim needs multiple seeds, the rigor contract, and
  ideally more than one model/layer. Do not over-read the numbers.
- **A 1B self-judge is weak.** Prefer the off-family Qwen judge; even then, a 3B
  judge on 1B outputs is pedagogy, not publication-grade evaluation.
- **"Low cosine" is not "orthogonal".** Cosine ~0.4 still shares meaningful
  overlap. The claim is that the directions are *distinguishable*, not
  independent — non-identifiability is a spectrum, and this lesson measures where
  on it these recipes land, not a binary.
- **The strongest test in the paper is not reproduced here.** The paper's
  headline construction uses explicit orthogonal perturbations `v + v⊥`; this
  lesson relies on naturally-arising low-cosine recipes instead. Faithful in
  spirit, but the *method* here stands on its own as a direct, falsifiable
  measurement regardless.
- **Matched alpha ≠ matched everything.** We match relative magnitude, but two
  directions could still differ in their coherence cost (gibberish rate). We log
  gibberish per direction so that confound is visible, not hidden.
- **Abliterated model.** On an aligned model there is nothing to re-install, so
  this specific "refusal-rate goes up" readout would not apply; the design would
  invert (strip refusal, measure compliance) as in lesson 10.

---

## 9. Repository

Source and full artifacts:
<https://github.com/dlmastery/steeringresearch/tree/master/steering_tutorials/non_identifiability>

See also
[lesson 1 — the probe (READ)](../hello_world/README.md) and
[lesson 2 — fixed-vector conditional steering (WRITE)](../hello_world_steering/README.md),
whose `model_utils`, `judge`, and diff-of-means recipe this lesson reuses
verbatim.
