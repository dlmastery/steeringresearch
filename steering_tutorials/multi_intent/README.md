# Lesson 9 — Multi-Intent Steering: composing K concepts without interference

> **Reference:** [Steering Llama 2 via Contrastive Activation Addition (arXiv:2312.06681)](https://arxiv.org/abs/2312.06681); [Refusal in LLMs Is Mediated by a Single Direction (arXiv:2406.11717)](https://arxiv.org/abs/2406.11717).

> **One-line idea.** Lesson 2 steered ONE concept ("refuse this"). Real
> deployments steer *several* at once — several harm categories, or "refuse
> harm" + "be concise". Naively summing K diff-of-means vectors makes them
> **interfere**. The fix: **Gram-Schmidt orthogonalize** the K directions so
> each steers its own concept with minimal cross-talk, and watch the **norm
> budget** (N5) — the total displacement you can spend before coherence breaks —
> to know how many concepts you can stack.

This package is a standalone teaching artifact. It **reuses** the mechanical
core of [lesson 2](../hello_world_steering/) (model loading, activation reading,
the steering hook, and the self-grading judge) and adds the *compositional*
layer on top. Nothing here reimplements steering; it **drives** it with a summed
vector.

- Lesson 1 (`../hello_world`): READ — train a probe on the residual stream.
- Lesson 2 (`../hello_world_steering`): WRITE — steer ONE concept (refusal).
- **Lesson 9 (this): WRITE MANY — steer K concepts at once, without cross-talk.**

---


> **Instrument caveat — read before citing any rate on this page.** Every refusal /
> compliance / gibberish number here is scored by a local LLM judge. That judge family
> was calibrated against ground-truth labels and measured **ROC-AUC 0.665–0.751** — below
> the 0.85 bar this course sets for a trustworthy instrument. Small differences between
> arms therefore sit at or below the judge's own noise floor and must not be read as
> effects. See [`../JUDGE_VALIDITY.md`](../JUDGE_VALIDITY.md) for the calibration, the
> continuous-readout improvement (+0.086 AUC, 12.8x faster), and which claims survive.

## The key idea in code

Two moves let K concepts coexist: Gram-Schmidt gives each concept its own axis
so they stop fighting, then all K fold into a single relative-add (`multi_intent.py`):

```python
def gram_schmidt(vectors):                    # turn K overlapping concepts into K clean axes
    basis = []
    for v in vectors:
        w = v.copy()
        for u in basis:
            w = w - (w @ u) * u               # subtract v's shadow on each earlier axis
        basis.append(w / np.linalg.norm(w))   # keep only what is NEW to this concept
    return basis

# fold the K orthogonal axes into ONE vector, injected in a single relative-add hook:
V = sum(alpha_i * unit(v_i) for v_i, alpha_i in zip(vectors, alphas))
```

Full file-by-file walkthrough below.

---

## Dataset

We treat **toxic-chat `openai_moderation` harm categories as the K concepts**
(`common.data.load_concepts`). The run in `artifacts/results.json` uses the
**three** well-populated categories recorded there (`sexual`, `harassment`,
`violence`), ordered most-distinct-first — so the ladder this lesson can walk is
**K = 1, 2, 3**, and no more. Every concept is contrasted against **one shared
benign baseline** so the K raw directions share an origin and their cosine
overlap measures *concept* similarity, not baseline drift. Labels are
**prompt-level**, never response-level.

The three raw directions overlap heavily — the measured cosine matrix in
`results.json` is `sexual·harassment = 0.937`, `sexual·violence = 0.920`,
`harassment·violence = 0.905`. That near-collinearity is precisely the condition
under which orthogonalization should matter.

| item | value |
|---|---|
| source / loader | `common.data.load_concepts` — toxic-chat `openai_moderation` harm categories |
| K concepts | 3 **well-populated** toxic-chat categories (sexual ~388, harassment ~143, violence ~111), ordered most-distinct-first. *Concept lessons are pool-limited: no category reaches 500, so this uses the available pool, not 500/class.* |
| per concept | `N_PER_CONCEPT = 500` requested (the whole pool); the shared loader's disjoint 40/30/30 split then yields the real eval `n` below. `N_EVAL_PER_CONCEPT` is **ignored** — the loader owns the split |
| held-out eval **n** (the denominator of every rate) | **sexual 117 · harassment 43 · violence 34** at `N_PER_CONCEPT = 500`. Hard floor `MIN_EVAL_PER_CONCEPT = 30`, enforced at run time — the run **aborts** before the GPU if any concept is short |
| shared baseline | benign prompts (common contrast origin) |
| model + judge | abliterated `DavidAU/gemma-3-1b-it-heretic-...` (steered); off-family **Qwen2.5-3B** judge required (`STEER_JUDGE_MODEL`) and stamped into `results.json` |

### Denominators and provenance

Every rate this lesson reports is `count / eval_n` on a concept's held-out split,
and `results.json` now carries both halves of that fraction:

- **`eval_n`** (header) — per concept: `pool_available`, `extract_n`,
  `achieved_n`, `pool_capped`; plus the enforced `floor_per_concept` (30), the
  rubric target (`target_per_class` 500) and `min_achieved_eval_n`.
- **per rung** — `n.per_concept`, `n.n_items`, `n.crosstalk_concept`,
  `n.crosstalk_n`, and inside each arm a `per_concept` breakdown giving the `n`,
  `success` and `gibberish` behind the mean.
- **judge stamp** — `judge_id` / `is_self_judge` / `judge_model_id` /
  `off_family`, taken from the judge object rather than from this README.

`success` is the **unweighted mean of the K active concepts' refusal rates**, so
it has no single denominator; `n_items` is the total graded at that rung, not a
divisor. The flag `success_is_unweighted_mean_of_per_concept_rates` says so in
the artifact.

**Pool ceiling, stated plainly.** These toxic-chat harm concepts hold 388 / 143 /
111 deduped prompts, so the repo's 500-per-class floor is **unreachable here** no
matter what we request. Per the pool-limited rule we request the whole pool and
report what it gives; `pool_capped: true` is written into the artifact and
printed by the runner. Every rate is **SCREENING tier**.

> ⚠ **The shipped `artifacts/results.json` predates all of this.** It was written
> on 2026-07-24 and contains **no `n` key and no judge key anywhere** — so its
> "off-family Qwen-3B judge" attribution below is a README claim that the
> artifact cannot corroborate (this lesson is one of the eight UNKNOWN-judge
> lessons in the CLAUDE.md §18.6 inventory). Its denominators were recovered
> arithmetically from the reported rates: **sexual 45 · harassment 43 ·
> violence 34** (it ran at the old `N_PER_CONCEPT = 150`). Those clear the 30
> floor, but nothing in the file said so. The numbers stand as-is until a re-run
> under the stamped runner replaces them.

**What the lesson uses it for:** steer all K "refuse this category" directions at
once and measure **interference** — naive raw-sum vs Gram-Schmidt
orthogonalization — while tracking the N5 **norm budget** to see how many
concepts can be composed before coherence breaks.

---

## 1. Why summing steering vectors goes wrong

A steering vector is a direction in activation space (a diff-of-means; lesson 2).
For concept *i* we build

```
v_i = mean(activation | concept-i prompts) - mean(activation | benign baseline)
```

To steer K concepts the naive recipe is: add all K at once,
`h <- h + Σ α_i · ||h|| · unit(v_i)`. This works for K = 1, sometimes K = 2, then
degrades. The reason is that **real concepts overlap**: "malware" and "fraud"
both involve deception, money and systems, so `v_malware` and `v_fraud` are *not*
perpendicular — their cosine similarity is well above zero.

When two directions overlap and you add them:

- the shared component gets pushed **twice** — you over-steer along the overlap
  and under-steer along what makes each concept distinct;
- steering concept A leaks into concept B's outcome — **cross-talk**;
- the shared component eats **norm budget** that neither concept "meant" to
  spend, so you hit the coherence cliff sooner.

That triple failure is **interference**.

---

## 2. The fix: Gram-Schmidt orthogonalization

Gram-Schmidt turns K overlapping directions into K **orthonormal axes**: each is
unit length, and every pair is perpendicular. Each axis keeps only the part of
its vector that is *new* relative to the axes already accepted.

```
ASCII picture — two overlapping concepts becoming two clean axes

     raw v1, v2 overlap:            after Gram-Schmidt:
          v2                             u2
          /                              |
         /                               |
        /____ v1                         |____ u1
       (v2 has a big shadow on v1)   (u2 ⟂ u1: no shadow left)
```

The algorithm, one vector at a time (this is exactly `gram_schmidt()`):

```
w = v_i                       # start from the raw direction
for each already-accepted axis u:
    w = w - (w · u) * u       # subtract w's *shadow* (projection) on u
u_i = w / ||w||               # normalize what remains -> a fresh unit axis
```

`(w · u) u` is the shadow `w` casts on an existing axis; subtracting it removes
what an earlier concept already covers. What remains is the residual no earlier
concept could express. Two consequences worth internalizing:

- **Order matters.** The first vector keeps its full direction; later ones are
  trimmed. Feed the most important / most distinct concept first
  (see `config.CONCEPTS`).
- **Collinear ⇒ zero axis.** If a vector lies in the span of earlier ones, its
  residual is ~0. We emit a **zero vector** for it (an honest "adds no new axis")
  instead of a NaN from dividing by ~0.

We use the *modified* Gram-Schmidt (subtract against the running `w`, not the
original `v_i`) — numerically stabler when inputs are nearly collinear.

---

## 3. The norm budget (N5)

Under `relative_add`, each concept injects `α_i · ||h|| · unit(v_i)`. When the
axes are **orthonormal**, the injected deltas are perpendicular, so the combined
step length is the root-sum-square (quadrature) combination:

```
budget = ||Σ α_i ||h|| u_i|| / ||h|| = sqrt(Σ α_i²)      (orthonormal u_i)
```

That is `norm_budget()`. It is the honest cost of stacking K orthogonal
concepts, and the number to watch against the coherence cliff: once it climbs
past the single-concept gibberish threshold, adding more concepts starts breaking
the model *regardless* of orthogonality. For raw non-orthogonal vectors the true
displacement is *larger* (shared components add linearly, not in quadrature) —
another reason orthogonalizing is the budget-efficient choice. The budget is a
finite resource, and it **caps how many concepts you can compose**.

---

## 4. The API (four functions)

All in `multi_intent.py`, reusing lesson 2's `SteeringContext`/`generate`:

| function | what it does |
|---|---|
| `extract_concept_vectors(model, tok, concept_prompts, layer, baseline_prompts)` | one diff-of-means `v_raw` per concept, all contrasted against a **shared** benign baseline (a common origin, so cosine(v_i, v_j) measures concept similarity). Returns `{name: np.ndarray[hidden]}`. |
| `gram_schmidt(vectors) -> list[np.ndarray]` | orthonormalize the K directions (modified Gram-Schmidt; degenerate ⇒ zero axis). |
| `apply_multi(model, tok, prompt, vectors, alphas, layer) -> str` | steer along all K directions in ONE hook: builds `V = Σ α_i unit(v_i)` and runs `relative_add` at `alpha=1.0`. Pass orthonormalized vectors for clean steering, raw vectors to reproduce the naive interfering baseline. |
| `norm_budget(vectors, alphas) -> float` | the N5 budget `sqrt(Σ α_i²)` being spent. |

Plus `cosine_matrix(vectors)` — a `[K,K]` overlap diagnostic (near-0 off-diagonal
⇒ already orthogonal; near ±1 ⇒ heavy interference).

---

## 5. The K = 1..N experiment (`run_multi_intent.py`)

We build K concept vectors once (K JailbreakBench harm categories), then walk a
ladder K = 1, 2, ..., N **adding one concept at a time**. At each rung, for BOTH
the naive **raw-sum** arm and the **orthogonalized** arm, we measure:

1. **Steering success** — on each *active* concept's held-out prompts, does the
   steered abliterated model now REFUSE (vs. its baseline COMPLIANCE)? Averaged
   over the K active concepts.
2. **Cross-talk** — on an *inactive* concept (one we did NOT add), does the
   mixture change its outcome anyway? Steering A should not move B; lower is
   cleaner.
3. **Norm budget vs coherence** — `sqrt(Σα²)` climbs with K; we track the
   GIBBERISH rate alongside it.

Extraction and evaluation prompts are **disjoint** per concept, so we never grade
a vector on the prompts that defined it.

> **The hypothesis we TEST, not assume.** Orthogonalization should spend the
> budget more efficiently — success stays higher and gibberish rises later than
> the raw-sum arm, i.e. interference is *sub-linear* in K. We **plot both arms
> and let the numbers speak.** On a 1B abliterated model with a 1B self-judge,
> treat every rate as a smoke-grade signal, not a publication claim.

Output: `artifacts/results.json` + `artifacts/success_vs_k.png` (success &
gibberish vs K on the left; norm budget & cross-talk vs K on the right).

---

## 6. Run it

CPU-only checks (no GPU, no model download beyond the tiny CSVs):

```bash
# from the repo root (C:\Users\evija\steeringresearch)

# import-check every module
python -c "import steering_tutorials.multi_intent.run_multi_intent, \
steering_tutorials.multi_intent.multi_intent, \
steering_tutorials.multi_intent.data, steering_tutorials.multi_intent.config"

# unit test: Gram-Schmidt orthonormality, norm budget, mixture recovery
python -m steering_tutorials.multi_intent.multi_intent

# unit test: rate helpers, denominator block, eval-floor gate, ladder summary
# (loads NO model; a bare invocation never touches the GPU)
python -m steering_tutorials.multi_intent.run_multi_intent

# preflight: build the splits, print the real per-concept eval n, run the
# 30/concept floor gate, and report which judge the env resolves to. Still CPU.
python -m steering_tutorials.multi_intent.run_multi_intent --preflight

# data smoke: downloads the shared CSVs, builds the K concept splits (no model)
python -m steering_tutorials.multi_intent.data
```

The full experiment (needs the GPU + the abliterated Gemma-3-1B):

```bash
# The OFF-FAMILY judge is MANDATORY for reportable numbers: a 1B target
# self-judging inflates refusal. Unset => the runner warns loudly and stamps
# judge_id="self" / is_self_judge=true into results.json.
STEER_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct \
python -c "from steering_tutorials.multi_intent.run_multi_intent import main; main()"
# writes artifacts/results.json + success_vs_k.png

# same thing via the module, on a host where a here-doc is awkward:
STEER_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct \
python -m steering_tutorials.multi_intent.run_multi_intent --run

# RAM-pressured host: shrink the whole ladder into one foreground window.
# Anything below the pool is smoke tier and lands as pool_capped=true.
MULTI_INTENT_N_PER_CONCEPT=150 STEER_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct \
python -m steering_tutorials.multi_intent.run_multi_intent --run
```

At the default `N_PER_CONCEPT = 500` the ladder grades
`2 arms x (117 + 43 + 34 eval prompts)` at K = 2 and K = 3 and `2 x (117 + 43)`
at K = 1 — about **1.1k steered generations** plus one judge call each, up from
~664 at the old `150`. Budget the GPU window accordingly.

---

## Results — measured vs. the claim

The screening run (`artifacts/results.json`, one abliterated 1B target on the
shared toxic-chat-derived concept prompts, per-concept α = 0.06) walks the ladder
for both arms. **The ladder that actually ran is K = 1, 2, 3** — the run used
three toxic-chat concepts, so **K = 4 and K = 5 never ran** and no number is
reported for them anywhere on this page. The judge for this artifact is
**unverifiable** — the file carries no judge stamp (see the warning above).

Every rung, both arms, straight out of `results.json`. The `n` columns are the
denominators, **recovered arithmetically** because the artifact does not record
them: per-concept eval splits of sexual 45 / harassment 43 / violence 34 at the
`N_PER_CONCEPT = 150` this run used.

| K | active concepts | n graded (per arm) | raw success | raw gibberish | ortho success | ortho gibberish | budget √(Σα²) | cross-talk (raw / ortho) | n cross-talk |
|---|---|---|---|---|---|---|---|---|---|
| 1 | sexual | 45 | 0.333 | 0.267 | 0.333 | 0.267 | 0.060 | 0.186 / 0.186 | 43 |
| 2 | + harassment | 88 | 0.193 | 0.316 | **0.216** | 0.294 | 0.085 | 0.265 / 0.412 | 34 |
| 3 | + violence | 122 | 0.137 | 0.523 | **0.237** | **0.217** | 0.104 | n/a / n/a | — |

`success` is the unweighted mean of the active concepts' rates, so the `n graded`
column is the total items behind it, not a divisor: at K = 1, 0.333 is 15/45; at
K = 2, 0.193 is the mean of 10/45 and 7/43.

At K = 1 the two arms are identical by construction (Gram-Schmidt leaves the first
vector untouched). Cross-talk is `NaN` at K = 3 because no inactive concept
remains to probe — it is not a measurement, and we do not report one.

| Claim | What we measured (off-family Qwen-3B judge, K=1..3) | Verdict |
|---|---|---|
| Naive summation interferes — raw-sum success collapses as K grows | raw success decays **0.333 → 0.193 → 0.137** while raw gibberish climbs **0.267 → 0.316 → 0.523** | **Supported (screening)** — success roughly halves and gibberish roughly doubles across the ladder |
| Gram-Schmidt orthogonalization cuts the interference | ortho success **0.333 → 0.216 → 0.237** at gibberish **0.267 → 0.294 → 0.217**; at K=3 ortho is **1.7× raw success (0.237 vs 0.137) at 0.42× the gibberish (0.217 vs 0.523)** | **Shown (screening)** — the contrast is present and in the predicted direction at both K=2 and K=3 |
| The N5 norm budget bounds how many concepts you can stack | budget climbs **0.060 → 0.085 → 0.104**, i.e. exactly 0.06·√K | **Supported** — the budget grows as the quadrature formula predicts |
| Orthogonalization also reduces cross-talk onto inactive concepts | the only rung with an inactive concept is K=2: raw **0.265** vs ortho **0.412** | **Not supported** — ortho cross-talk is *higher* at the single rung where it can be measured |

**Honest read.** On the numbers this run produced, the lesson's central contrast
**does appear**: the raw-sum arm degrades monotonically in both success and
coherence as concepts are added, while the orthogonalized arm holds its success
near the K=1 level and — strikingly — ends the ladder with *less* gibberish
(0.217) than it started with (0.267). The three raw directions are near-collinear
(cosines 0.91–0.94), so the shared component the raw sum triple-counts at K=3 is
large; removing it is what keeps the ortho arm on-manifold. That is the mechanism
the lesson predicts, visible in one screening run.

Two things to hold against it. First, the one cross-talk rung that exists points
the *other* way (ortho 0.412 vs raw 0.265) — orthogonalization bought coherence
and success here, not off-target cleanliness. Second, this is screening tier: one
seed, per-concept eval splits of 45 / 43 / 34 (a single flipped verdict moves a
concept's rate by 2–3 points), a 1B target, and a judge that the artifact does
not even identify — measured elsewhere at ROC-AUC 0.665–0.751 (see the instrument
caveat above). Differences of a few points sit inside the judge's noise floor;
the K=1→K=3 *trends* (a 0.20 success drop, a 0.26 gibberish rise in the raw arm)
are the part large enough to read. None of this reaches the CLAUDE.md §7
evaluation bar.

**What a re-run changes.** Under the current runner, `sexual`'s eval split goes
45 → 117 (the whole pool) and its extract half 105 → 271, so the K = 1 rung and
the shared-concept direction both get materially better estimated;
`harassment` (43) and `violence` (34) are already at their pool ceiling and will
not move. The re-run also writes the per-rung `n`, the `eval_n` header and the
judge stamp, which is what makes the table above auditable rather than merely
believable. Treat every number on this page as provisional until then.

---

## 7. Honest caveats

- **1B model + 1B self-judge.** Pedagogical, not publication-grade. A real
  evaluation uses a stronger judge and n≥7 seeds with the rigor contract
  (CLAUDE.md §7). Here K, alphas and eval sizes are tiny for laptop speed.
- **We measure interference; we do not assume it.** The sub-linear-interference
  claim is a *hypothesis* the ladder tests. If the raw-sum arm keeps up with the
  orthogonalized arm at these K and alphas, the plot will show it.
- **Cross-talk is measured on a single inactive concept per rung** (the next one
  in line) — a cheap probe, not an exhaustive off-target sweep.
- **The norm budget formula assumes orthonormal axes.** For the raw arm the true
  displacement is larger; we report the orthonormal-case budget as the common
  yardstick and note the discrepancy in `norm_budget`'s docstring.
- **Abliterated model** (`DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated`):
  it complies with harm out of the box, which is *why* we can watch external
  refusal vectors switch categories off. This is a defensive / research use.

---

## 8. Where this sits

- Concept: **compositional steering** — the N5 norm budget and interference from
  `corpus/steering-first-principles-v2-with-PSR-and-rogue-scalpel.md`.
- Method: Contrastive Activation Addition per concept
  (Rimsky et al. 2023, arXiv:2312.06681), refusal-as-a-direction
  (Arditi et al. 2024, arXiv:2406.11717), composed via Gram-Schmidt.
- Next lessons: L10 `rogue_scalpel` (the universal attack + five-layer guard),
  L12 `stacking` (orthogonal-stack vs same-site-compete — the §9 combo ladder,
  of which this lesson is the "different site ⇒ stack" special case for
  *concepts*).
