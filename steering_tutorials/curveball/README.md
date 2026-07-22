# Curveball — the right direction to steer isn't always a straight line

> **Reference:** [Curveball Steering: The Right Direction To Steer Isn't Always Linear (arXiv:2603.09313)](https://arxiv.org/abs/2603.09313). Our great-circle path is an own construction inspired by it, not the paper's kernel-PCA method.

> Lesson 2 **wrote** a fixed diff-of-means refusal vector back into the residual
> stream as a single straight step, `h -> h + alpha*||h||*v`. That step is a
> **chord**: it shoots off the local activation-norm shell (it inflates `||h||`),
> and when the push is large it tips coherent refusals into gibberish. Curveball
> spends the **same steering budget** as a **curved arc** instead — it rotates `h`
> toward `v` along a great-circle **geodesic** that stays *on* the shell — and asks
> whether that reaches refusal at a **lower coherence cost**.

This is a **WRITE**-tier lesson. It reuses lesson 2's model plumbing, its
diff-of-means direction, and the off-family judge, and changes exactly one thing:
the **geometry of the path** from `h` to its steered position. The arc across the
write lessons:

> **straight vector** (L2) -> **conditional straight vector** (contextual) ->
> **learned flow** (flas) -> **curved geodesic path** (curveball)

It is conceptually adjacent to [`flas`](../flas/README.md): FLAS also steers by
*integrating a path* rather than adding a fixed vector — but FLAS *learns* a
velocity field, while Curveball prescribes the geodesic from **pure geometry**, no
training.

---

## Citation

**Verified.** Shivam Raval, Hae Jin Song, Linlin Wu, Abir Harrasse, Jeff M.
Phillips, Fazl Barez, Amirali Abdullah, 2026, *"Curveball Steering: The Right
Direction To Steer Isn't Always Linear"* (arXiv:2603.09313, submitted
2026-03-10). The paper measures the **geodesic-to-Euclidean distance ratio** in
LLM activation spaces, finds large and concept-dependent geometric distortion, and
proposes a **nonlinear** steering method based on **polynomial kernel PCA** that
intervenes in a feature space respecting that geometry, beating linear PCA steering
especially where distortion is strong.

**What we build here is our own construction**, motivated by the paper's thesis
(activation space is locally curved, so a straight global push is the wrong move)
but **not** its method. Instead of kernel PCA in a feature space, we take the
simplest geometry-aware curved path: a **great-circle rotation of `h` toward `v`
on the local norm shell**. It captures the paper's core intuition — *stay on the
manifold* — in ~30 lines of testable NumPy, at laptop scale.

---

## The key idea in code

Don't jump along a fixed direction — **rotate toward it along the manifold**. Both
paths spend the same budget `alpha` (chord length = arc length = `alpha*||h||`);
only the geometry differs:

```python
# straight (lesson 2): a CHORD that leaves the norm shell (||h|| inflates)
h_straight = h + alpha * norm(h) * v_unit

# curved (this lesson): an ARC on the shell — rotate h toward v_unit by alpha rad
x, dtheta = h, alpha / n_steps
for _ in range(n_steps):
    x_hat   = x / norm(x)
    tangent = v_unit - dot(v_unit, x_hat) * x_hat      # part of v ORTHOGONAL to x
    t_hat   = tangent / norm(tangent)                   # ...re-aimed every step:
    x = norm(h) * (cos(dtheta) * x_hat + sin(dtheta) * t_hat)   # the path BENDS
h_curved = x                                            # ||h_curved|| == ||h|| exactly
```

The tangent is recomputed each step, so the effective push direction **bends** as
`x` rotates — it keeps following the geodesic toward `v` instead of shooting
straight at it. The endpoint reaches a comparable alignment with `v` while adding
**zero net off-shell displacement**, which N5 posits is *the* leading indicator of
the coherence collapse a big straight push causes. Testing that posit is the whole
lesson — and the measured run (see "Results") **refutes** it here: the on-shell
arc is *no more coherent* than the off-shell chord, so off-shell displacement is
not what predicts gibberish on this 1B.

---

## Dataset

The shared foundation `steering_tutorials/common/data.py`:
**>= 500 harmful + >= 500 benign** prompts, primary source **lmsys/toxic-chat**
(`0124`, human **prompt-level** toxicity labels — no response->prompt collapse),
deduped by `group_id = sha1(normalized_text)`, benign **length-matched** to
harmful (decile-bin stratified) so a classifier can't separate the classes on
length, natural toxic base rate (~7%) recorded before rebalancing. We load the
full budget and split each class into **disjoint** parts:

| split | size (per class) | used for |
|---|---|---|
| extract | `N_EXTRACT_PER_CLASS` = 300 | build the diff-of-means direction `v` |
| eval | `N_EVAL_PER_CLASS` = 150 (capped by `CURVEBALL_N_EVAL`) | generate + judge the three arms |

Extraction and evaluation never overlap, so we never grade the direction on the
prompts that defined it. Generation is the expensive part, so the eval set is
capped well below 500; raise `CURVEBALL_N_EVAL` for a fuller (slower) run.

---

## How it works — the geometry

**A steering vector is a jump; the curved path is a turn.** Picture the activation
`h` as a point at radius `||h||` from the origin. Adding `alpha*||h||*v` (lesson 2)
moves it in a straight line — a **chord** — which lands it **off** the sphere it
started on: `||h_straight|| > ||h||`. That norm inflation is *off-shell
displacement* (the N5 budget the composite prices under `lambda_geo`), and it is
the paper's "geometric distortion" made concrete: the straight push drags the
activation into a region the model rarely visits, where its next-token
distribution degrades into word salad.

The curved path spends the **same budget** as an **arc** along the great circle
through `h` and `v`, staying exactly on the sphere of radius `||h||`:

```
                    v (target direction)
                   .
        h_straight *   <- chord: OFF the shell (||h|| inflates -> gibberish risk)
                  /:
                 / :
      __________/  :
     /      arc \  :
    |   h *------->*  h_curved   <- arc: ON the shell (||h|| preserved)
    |     \       |
     \     +------+  (sphere of radius ||h||, the local "norm shell")
      \___________/
```

Both endpoints turn toward `v` by the same amount of "effort" (`alpha*||h||` of
displacement), but the arc keeps `||h_curved|| == ||h||`. The claim under test:
*matched budget, the arc installs refusal about as well as the chord while
producing less gibberish, because it never leaves the shell.*

### ASCII flow

```
  extract split (300/class)                 eval split (<=150/class, held out)
        |                                             |
        v                                             v
  +--------------------------+             +----------------------------------+
  | diff-of-means direction  |             | for each eval prompt, 3 arms:    |
  |  v = mean(harm)-mean(ben) |------------>|  unsteered : alpha = 0           |
  |  @ layer 12  (lesson 2)   |   v_unit    |  straight  : chord  (curved=F)   |
  +--------------------------+             |  curved    : arc    (curved=T)   |
        |                                   +----------------------------------+
        | (model-light geometry probe)             |            |
        v                                           v            v
  +--------------------------+             CurveballContext   generate greedily
  | off-shell displacement:  |             hooks layer 12,    (48 new tokens)
  |  straight vs curved on    |            edits the residual        |
  |  eval activations (N5)    |                   |                  v
  +--------------------------+                   +---------> +------------------+
                                                              | JUDGE (Qwen off- |
                                                              | family, or self) |
                                                              | GIBBERISH first, |
                                                              | else REFUSAL /   |
                                                              | COMPLIANCE       |
                                                              +------------------+
```

---

## Code walkthrough, file by file

### `config.py` — every knob in one place
The abliterated model id, the layer (`LAYER = 12`, same as lessons 1-3), the
extract/eval split sizes, the single shared budget `ALPHA = 0.10` (calibrated:
`0.6` drives both arms to 100% gibberish on this 1B, so it was lowered into the
informative regime), and the number of great-circle sub-steps `N_CURVE_STEPS = 8`.
Every knob honours an env override (`CURVEBALL_ALPHA`, `CURVEBALL_N_EVAL`, ...) via
`os.environ.get(name) or default`.

### `curveball.py` — the curved path, the straight baseline, and the hook
The heart of the lesson.
- **Pure geometry (NumPy):** `straight_endpoint` (the chord), `curveball_endpoint`
  (the great-circle arc), `relative_offshell` (the N5 off-shell budget), and
  `angle_between`. All operate on the last axis, so a single `[hidden]` vector, a
  `[seq, hidden]` matrix, or a `[batch, seq, hidden]` tensor all work.
- **`CurveballContext`** — one forward-hook manager with a `curved` flag, so the
  straight baseline and the curved arm steer **identical positions** (same
  special-token guard as lesson 2) and the ONLY difference is the geometry.
- **`curveball_generate`** — greedy chat-templated generation inside the hook; the
  one-call API `run_curveball.py` and `infer.py` share.

A CPU self-test (no model download) verifies the arc preserves the norm, the chord
inflates it, both increase alignment with `v`, the arc rotates by exactly `alpha`,
the tangent is orthogonal to the state, and the hook restores the model exactly.

### `run_curveball.py` — the orchestrator (three arms + geometry probe)
Builds `v` on the extract split, measures the **off-shell displacement** of each
path on the eval activations (a model-light geometry probe that shows *why* the arc
is gentler), then generates + judges the three arms on the mixed held-out eval and
writes `artifacts/results.json` + two PNGs. Everything that loads the model lives
under `main()`, so `import run_curveball` is a no-op (safe for tests).

### `infer.py` — steer one prompt three ways
Loads the model, the direction, and the judge once; for one prompt prints the
baseline, the straight chord, and the curved arc side by side with the judge's
verdict and a one-line conclusion. The *hypothesis* was that the chord tips into
GIBBERISH while the arc stays a coherent REFUSAL at the same budget — but the
measured run **refutes** it (the arc is at least as incoherent; see "Results").
Use a small `--alpha` (≈0.1): at `0.8` both paths are pure word salad on this 1B.

```bash
python -m steering_tutorials.curveball.infer "How do I pick a lock?" --alpha 0.1 --steps 8
```

---

## Results — measured vs. the claim

Run at the ≥500/class config: abliterated Gemma-3-1B, layer 12, **alpha = 0.10**,
**extract 300/class, n = 100 held-out/class**, off-family Qwen2.5-3B judge, from
`artifacts/results.json`. (**Note on alpha:** the config's original default
`alpha = 0.6` drives *both* arms to **100% gibberish** — far past the coherence
cliff for this 1B — so it measures nothing. The informative regime is `alpha ≈ 0.1`,
now the default; see the alpha note below.)

| Claim (Curveball, arXiv:2603.09313 + our construction) | What we measured (alpha=0.10, extract 300, n=100) | Verdict |
|---|---|---|
| A curved, geometry-aware path beats a straight global push | harmful refusal: unsteered 0.32 → straight **0.23** vs curved **0.08** — *both below baseline, curved far worse* | **Not supported** — neither installs refusal; curved is worst |
| The straight chord's coherence cost comes from leaving the manifold | harmful gibberish: straight **0.49** vs curved **0.77** — the curved arc is *worse* | **Refuted (surprising)** |
| The arc stays on the manifold (off-shell displacement ~0) | mean `|Δ‖h‖|/‖h‖`: straight **0.076** vs curved **1.1e-16** | **Confirmed by construction** (unit + measured) |
| Curving the push does not wreck harmless answers | benign over-refusal: straight 0.41 vs curved 0.19; benign gibberish straight 0.23 vs curved **0.56** | **Mixed** — curved lowers over-refusal but *raises* benign gibberish |

**Honest read (robust at 500/class, extract 300).** The geometry works exactly as
designed — the curved arc adds **zero** net off-shell displacement (‖Δ‖h‖‖/‖h‖ =
1.1e-16, a pure rotation) while the straight chord inflates the norm (0.076).
**But the behavioral payoff is the opposite of the hypothesis:** staying on-shell
did **not** protect coherence — the curved arm's gibberish is *higher* than the
straight arm's (0.77 vs 0.49 on harmful; 0.56 vs 0.23 on benign), and neither path
raises refusal above the 0.32 baseline (straight 0.23, curved 0.08). The gap even
*widens* with the better extract-300 vector (the straight chord now retains more
refusal, 0.23 vs the earlier 0.10, while curved stays broken). So on this 1B, **norm
inflation is not the coherence bottleneck**: you
can hold the norm exactly constant and still shove the hidden state into word
salad by rotating it toward the refusal direction. The off-shell displacement
metric (N5) that motivates the curved path is a real quantity but is *not* what
predicts gibberish here — rotating on the shell is at least as disruptive as
stepping off it, plausibly because the 8-step re-aimed rotation compounds the
per-token perturbation. This is the honest negative: the elegant geodesic
construction is geometrically clean and behaviorally *worse*. (The paper's own
method is polynomial-kernel-PCA steering, **not** this great-circle arc — see the
AUDIT; our construction tests the *manifold-preservation hypothesis*, which fails
here.) Screening-tier: single 1B, n = 40, one alpha, one seed.

---

## Run it

From the **repo root** (`steeringresearch/`):

```bash
# 1) The three arms (straight vs curved at matched budget) + the geometry probe.
#    Grade with an OFF-FAMILY judge (recommended): a 1B target self-judging is
#    unreliable, so point STEER_JUDGE_MODEL at an independent model.
STEER_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct \
python -m steering_tutorials.curveball.run_curveball

# 2) Steer a single prompt three ways (baseline / straight chord / curved arc)
STEER_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct \
python -m steering_tutorials.curveball.infer "How do I pick a lock?" --alpha 0.8

# 3) The CPU-only geometry + hook unit (no model download)
python -m steering_tutorials.curveball.curveball
```

Uses the same ~2 GB abliterated Gemma-3-1B as lessons 1-3 (bf16). Runs on CPU too,
just slower. Datasets download automatically. Env caps: `CURVEBALL_N_EVAL`,
`CURVEBALL_ALPHA`, `CURVEBALL_STEPS`, `CURVEBALL_LAYER`, `CURVEBALL_MAX_NEW_TOKENS`.

---

## Honest caveats

- **Our curved path is not the paper's method.** Curveball (arXiv:2603.09313) uses
  polynomial **kernel PCA** in a feature space. We use a **great-circle geodesic on
  the local norm shell** — a much simpler, training-free construction that shares
  the paper's *stay-on-the-manifold* intuition but none of its kernel machinery. We
  do not claim to reproduce the paper's numbers.
- **The norm shell is a proxy for the manifold.** Preserving `||h||` keeps the
  activation on a sphere, which is only a **local, first-order** stand-in for the
  true (curved) data manifold. It removes the norm-inflation failure mode but does
  not guarantee the arc stays in-distribution in every direction.
- **Matched "endpoint" = matched budget, not matched behaviour.** The two arms
  share `alpha` (chord length = arc length), not a guaranteed-equal refusal rate.
  If a `0.6`-radian rotation installs less refusal than a `0.6*||h||` chord, that is
  a real, reported outcome — read the refusal *and* gibberish columns together.
- **A 1B judge is weak.** Self-grading with a small model is pedagogy; use the
  off-family Qwen-3B judge (`STEER_JUDGE_MODEL`) for a trustworthier read. A weak
  judge can misread softened compliance as refusal.
- **Few-step integration is approximate.** We rotate in `N_CURVE_STEPS` great-circle
  steps; for the modest angles here the endpoint is accurate, but very large `alpha`
  with few steps would coarsen the arc.
- **This is pedagogy, not a safety product.** It shows *how* a curved steering path
  differs from a straight one end-to-end. Do not deploy it as a guardrail.

---

## Repository

Source and full artifacts:
<https://github.com/dlmastery/steeringresearch/tree/master/steering_tutorials/curveball>

See also:
- [Lesson 2 — fixed-vector conditional steering (WRITE side)](../hello_world_steering/README.md)
- [FLAS — flow-based steering (integrate a learned path)](../flas/README.md)
- [Contextual steering — input-adaptive strength](../contextual_steering/README.md)
