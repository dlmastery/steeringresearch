# Lesson 9 — Multi-Intent Steering: composing K concepts without interference

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

## Dataset

We treat **JailbreakBench harm categories as the K concepts** (`data.py`,
`load_multi_intent`). JailbreakBench (Chao et al. 2024, arXiv:2404.01318) ships
`harmful-behaviors.csv` with a `Category` column — 10 categories × 10 prompts.
We pick **four** distinct categories as concepts (`config.CONCEPTS`):
Malware/Hacking, Fraud/Deception, Harassment/Discrimination, Physical harm.
Every concept is contrasted against **one shared benign baseline** (40 prompts
from `benign-behaviors.csv`) so the K raw directions share an origin and their
cosine overlap measures *concept* similarity, not baseline drift. Labels are
**prompt-level** (the `Goal` column), never response-level.

| item | value |
|---|---|
| source / loader | JailbreakBench `Goal`/`Category` via `hf_hub_download` (`data.load_multi_intent`) |
| K concepts | 4 JBB harm categories, ordered most-distinct-first |
| per concept | 10 prompts → 5 extract (build the vector) / 5 eval (disjoint, held out) |
| shared baseline | 40 benign prompts (common contrast origin) |
| model + judge | abliterated `DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated`, self-graded |

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

# unit test: rate helpers + ladder summary
python -m steering_tutorials.multi_intent.run_multi_intent

# data smoke: downloads JBB CSVs, builds the K concept splits (no model)
python -m steering_tutorials.multi_intent.data
```

The full experiment (needs the GPU + the abliterated Gemma-3-1B):

```python
from steering_tutorials.multi_intent.run_multi_intent import main
main()   # writes artifacts/results.json + success_vs_k.png
```

---

## Results — measured vs. the claim

The screening run (`artifacts/results.json`, K = 1..4, one abliterated 1B target
self-graded, ~5 held-out prompts per concept) walks the ladder for both arms:

| Claim | What we measured (screening) | Verdict |
|---|---|---|
| Naive summation interferes — raw-sum success collapses as K grows | raw success 0.40 (K=1) → 0.10 (K=2) → 0.13 (K=3) → 0.15 (K=4); gibberish 0.60 → 0.90 → 0.87 → 0.85 | Supported — raw-sum success falls and gibberish dominates once K > 1 |
| Gram-Schmidt orthogonalization cuts the interference | ortho success 0.40 / 0.30 / 0.00 / 0.40 vs raw 0.40 / 0.10 / 0.13 / 0.15 | Directionally supported — ortho beats raw at K=2 (0.30 vs 0.10) and K=4 (0.40 vs 0.15); K=3 lands the wrong way (0.00) |
| The N5 norm budget bounds how many concepts you can stack | budget = sqrt(Σα²) climbs 0.060 → 0.085 → 0.104 → 0.120, monotone in K | Supported — the budget grows exactly as the quadrature formula predicts |

Read this as a smoke-grade signal, not a result: with K held-out sets of ~5
prompts and a 1B self-judge, single rungs swing on noise — the K=3 orthogonalized
arm collapsing to 0.00 success (below the raw arm) is the clearest sign of it. The
*shape* the claim predicts is visible — raw-sum interference rising with K,
orthogonalization spending the budget more efficiently at most rungs, and the norm
budget climbing as sqrt(Σα²) — but the separation is neither clean nor monotone at
1B. A publication claim needs a stronger off-family judge, larger eval sets, and
n≥7 seeds with the rigor contract.

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
