# METHODOLOGY.md — exactly how every experiment works (the transparent recipe)

> **Purpose.** This is the single, authoritative, code-accurate description of
> *how a steering experiment is run* in this project: how a steering vector is
> generated, how it is injected, what is measured, and how a verdict is reached.
> Every per-hypothesis design doc has a **"Pseudocode & Methodology"** section
> that specializes this recipe; this file is what they all reference.
>
> Nothing here is trained (except the three auxiliary-component hypotheses E15/
> E45/E20 — see [`TRAINING-PROCESS.md`](TRAINING-PROCESS.md)). The base model is
> **frozen**; a steering vector is *estimated* from activations in closed form
> and *injected* at inference time.

Every pseudocode block below names the **real function** that implements it, so
you can read the source directly. Core modules: `src/steering/extract.py` (vector
generation), `hooks.py` (injection), `eval.py` (the five axes + composite),
`geometry.py` (geometry probes), `controls.py` + `stats.py` (controls + rigor),
`judge.py` (the off-family behavior judge), `runner.py` / `scripts/*` (drivers).

---

## 1. How a steering vector is generated

A steering vector is a single direction in the residual stream that, when added
to a layer's hidden state, pushes the model toward a target behavior. It is built
from **contrast pairs** — short texts that do vs. do not exhibit the behavior.

### 1.1 Inputs

```
pairs = [(pos_text, neg_text), ...]     # e.g. ocean: ("The waves crashed...", "The tax form...")
layer L                                  # which residual block to read/inject
```
Contrast pairs live in `src/steering/data/concepts_multi.json` (ocean, happiness,
anger, formality) and `axbench_mini.json`; loaded by `datasets.load_concept`.

### 1.2 Collect activations (one forward pass per text — NO gradients)

```
# extract.collect_activations  (cached by extract.collect_activations_cached)
for (pos, neg) in pairs:
    H_pos = forward(model, pos).hidden_state[L]      # [seq, dim]
    H_neg = forward(model, neg).hidden_state[L]
    h_pos += [ mean_over_tokens(H_pos) ]             # [dim], mean-pooled
    h_neg += [ mean_over_tokens(H_neg) ]
# h_pos, h_neg : [n_pairs, dim]
```
Activations are cached to disk once and reused across the whole ladder
(`collect_activations_cached`); the cache key includes model id + quantization +
the exact contrast-pair text.

### 1.3 Build the direction (closed form — pick ONE source)

```
# extract.diffmean_vector  — the default "DiffMean" source
v_diffmean = mean(h_pos, axis=0) - mean(h_neg, axis=0)        # one subtraction

# extract.pca_top1_vector  — the "PCA" source
D          = h_pos - h_neg                                    # per-pair differences [n, dim]
v_pca      = top_right_singular_vector(SVD(D))                # ONE SVD, not gradient descent
v_pca      = sign_align(v_pca, v_diffmean)                    # make cosine comparable

# extract.fisher_ratio  — used ONLY to choose the layer, not to steer
w          = unit(v_diffmean)
fisher[L]  = (mean(h_pos·w) - mean(h_neg·w))**2 / (var(h_pos·w) + var(h_neg·w))
best_layer = argmax_L fisher[L]                               # extract.best_layer
```

`extract.build_vector_bank` returns, per layer, `{diffmean, pca, cosine_dm_pca,
fisher}`. There is **no optimizer, loss, or epoch** anywhere in this path.

### 1.4 Optional normalization

```
if normalize:  v = v / ||v||      # makes alpha a source-comparable displacement (E7/E36)
```

---

## 2. How a vector is injected (the OPERATION)

The vector is added into layer `L` for every non-special token position via a
forward hook (`hooks.SteeringContext`, `hooks.apply_operation`). `alpha` is the
strength knob. Special tokens (BOS, turn markers) are never steered
(`hooks.build_position_mask`). Four operations:

```
# hooks.apply_operation(h, v, operation, alpha)   h = residual state [..., dim]
add          :  h' = h + alpha * v                              # CAA / ActAdd (classic)
relative_add :  h' = h + alpha * ||h|| * unit(v)                # E7: alpha = FRACTION of ||h||
project_out  :  h' = h - alpha * (h · unit(v)) * unit(v)        # ablate the direction (guard)
rotate       :  e1 = unit(h);  e2 = unit(v - (v·e1) e1)         # Gram-Schmidt in the (h,v) plane
                h' = ||h|| * (cos(alpha)*e1 + sin(alpha)*e2)     # norm-preserving rotation
```

`relative_add` is special: it normalizes to a unit direction and scales by
`alpha*||h||`, so the displacement magnitude is **identical regardless of the
vector**, and `alpha` is a fraction of the residual norm (scale-comparable).

---

## 3. What is measured (the five axes + geometry)

For a given (concept, layer, source, operation, alpha), the harness measures
**steered vs. unsteered** on five axes (`eval.evaluate_bundle`):

```
1. BEHAVIOR efficacy   : does the steered text exhibit the target behavior?
     - rung >=3 / external: eval via the OFF-FAMILY JUDGE (judge.GeminiJudge) — see §5
     - cheap screening    : concept_rate(text, lexicon)  [PROXY — never backs a claim]
2. CAPABILITY retention : MMLU multiple-choice accuracy delta (eval.mcq_accuracy) — ground truth
3. COHERENCE           : perplexity delta on WikiText (eval.perplexity) — from logits
4. SAFETY integrity    : JailbreakBench compliance rate on harmful prompts generated UNDER steering
5. SELECTIVITY         : harmless-input refusal rate (over-refusal dual)
GEOMETRY probes (always): offshell ||Δh|| (geometry.offshell_displacement),
     angular 1-cos (angular_displacement), effective-rank drop, norm budget.
```

### 3.1 The composite (Goodhart-resistant; fingerprint `a9001e87087e`)

```
# eval.COMPOSITE_FORMULA
composite = behavior
          - λ_cap  * max(0, MMLU_drop_pp)
          - λ_coh  * max(0, ΔPPL_norm)
          - λ_safe * compliance_rate
          - λ_sel  * max(0, harmless_refusal_rate)
          - λ_geo  * max(0, offshell_displacement)
```
A method cannot win by sacrificing one axis (incoherence/leak/over-refusal are
priced). Reported to 4 dp alongside every axis separately.

---

## 4. Controls and statistical rigor (confirmation rungs)

Cheap screens are single-seed on the proxy. A **claim** requires controls + the
four-part contract (`controls.py`, `stats.py`):

```
# CONTROLS at the SAME displacement (only DIRECTION differs) — controls.py
v_random   = matched_norm_random(v)        # random unit direction, matched magnitude
v_shuffled = shuffled_label_vector(pos,neg)# DiffMean of a random re-partition (labels destroyed)
stability  = extraction_stability(pos,neg) # bootstrap cosine of v to full-data v  (gate >0.85)

# FOUR-PART CONTRACT across n>=20 seeds — stats.rigor_report
for seed in 1..N:
    beh[cond][seed] = mean_over_prompts( score( steered_generate(cond, seed) ) )
external_ready = (paired_wilcoxon(real, control).p < 0.05)
              and bootstrap_ci(real-control) excludes 0
              and holm_bonferroni(family p-values) rejects
              and ordinal_gate(worst real seed > best control seed)
```
See [`FINDINGS.md`](../FINDINGS.md) S-15 for a worked example (E7).

---

## 5. The behavior judge (off-family, validated)

```
# judge.GeminiJudge (Gemma generator, Gemini judge -> no same-family circularity)
score = judge.score(text, behavior_name, behavior_description)   # {behavior 0-10, coherence 0-10}
behavior_efficacy = score.behavior / 10                          # in [0,1]
```
Temperature 0, JSON output, disk-cached, 429-backoff, `JudgeUnavailable` →
falls back to the proxy. Validated: real ocean prose 8/10, off-topic 0/10,
keyword-soup 2/10 (not fooled by keyword stuffing). Validate against a reference
set with `judge.calibration_agreement` before trusting it.

---

## 6. The benchmark ladder (where each rung spends compute)

```
Rung 0 UNIT   (sec)   : vector changes logits; state restores exactly      (tests/)
Rung 1 SMOKE  (1-3m)  : monotone effect + bounded PPL + no safety leak
Rung 2 DEV    (10-20m): beats baseline on held-out concepts at matched coherence
Rung 3 STANDARD(1-3h) : Pareto-dominates prior method; controls + n>=20 + judge
Rung 4 FULL   (half-d): full multi-axis win + ablations + red-team neutralized
```
A method clears rung *k*'s gate before spending rung *k+1* compute.

---

## 7. The standard experiment loop (what a driver actually does)

```
# scripts/campaign_sweep.py / runner.run_single_experiment (screening),
# scripts/confirm_e7.py (controlled confirmation)
bank   = extract_bank(model, tokenizer, load_concept(concept), layer)   # §1
v      = bank[layer][source]                                            # diffmean | pca
for (layer, alpha, operation, seed) in sweep_grid:                      # the ONE thing varied
    metrics = evaluate_bundle(steered vs unsteered across the 5 axes)   # §3
    log(experiment_log.jsonl, config + metrics + composite + fingerprint)
verdict = KEEP if composite beats champion at matched coherence else DISCARD
```
Each experiment changes **exactly one** knob from the champion (the Karpathy
single-axis rule). Every run authors a pre-registered 7-step reasoning entry
before launch (`reasoning_annotations.json`).

---

## 8. Reading a per-hypothesis "Pseudocode & Methodology" section

Each design doc specializes the above with: (a) the **vector recipe** that
hypothesis uses (which source/layer/construction), (b) the **exact procedure**
(what is swept, what is compared), (c) the **measurement + decision rule** (the
metric and the pre-registered falsifier threshold). When a hypothesis needs
machinery that does not exist yet (CAST gating, SAE, hypernetwork, multi-vector
orchestration), the section says so — that missing code is why it is UNTESTED.
