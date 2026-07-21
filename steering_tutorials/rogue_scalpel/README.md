# Rogue Scalpel — red-teaming a steering intervention, and defending it

> **Reference:** [Rogue Scalpel — red-teaming activation steering (arXiv:2509.22067)](https://arxiv.org/abs/2509.22067); steering vector via [Contrastive Activation Addition (arXiv:2312.06681)](https://arxiv.org/abs/2312.06681).

> Lesson 1 built a probe that **reads** "is this harmful?" out of Gemma-3-1B.
> Lesson 2 **writes** back: adding the refusal direction `+v` to the residual
> stream *installs* refusal. Lesson 10 turns that scalpel around. The **same**
> residual-stream write can be inverted to **strip** refusal — a jailbreak. We
> (a) demonstrate that attack, then (b) implement and ablate a layered **guard**
> and show it drives the attack-success-rate back down.

Steering is **dual-use**. This is the uncomfortable core of activation steering
as a safety tool: the mechanism that adds safety can remove it. This lesson makes
that concrete on one small local model, and then does the responsible thing —
builds a defense and measures whether it holds.

> **Defensive / educational only.** The "attack" here is a red-team probe against
> *our own* intervention on a 1B toy model. It contains no operational exploit
> content; the harmful prompts are the JailbreakBench placeholders reused from
> lesson 2. The purpose is to prove a guard neutralizes it. See
> [CLAUDE.md section 10](../../CLAUDE.md) (the Rogue-Scalpel mandate).

---

## The key idea in code

Steering is dual-use in one sign flip: the same refusal direction lesson 2
*adds* can be *removed* to jailbreak the model — and a norm clamp bounds that
edit back toward the manifold (`attack.py` + `guard.py`):

```python
# ATTACK — strip refusal out of the residual stream (attack.py)
def suppress_refusal_delta(h, u, mode):
    if mode == "project_out":
        return -(h @ u) * u          # delete the refusal component entirely
    return -alpha * norm(h) * u      # or push AGAINST it: lesson 2's add, sign-flipped

# GUARD — clamp the adversarial edit back toward the manifold (guard.py)
factor = min(1, budget * norm(h_clean) / norm(delta))   # bound ||Δh|| <= budget·||h||
h = h_clean + delta * factor
```

Full file-by-file walkthrough below.

---

## Table of contents

1. [The dual-use idea](#1-the-dual-use-idea)
2. [The attack](#2-the-attack)
3. [The guard: three layers](#3-the-guard-three-layers)
4. [Attack vs. guard, in one picture](#4-attack-vs-guard-in-one-picture)
5. [The ablation ladder](#5-the-ablation-ladder)
6. [Code walkthrough, file by file](#6-code-walkthrough-file-by-file)
7. [Results](#7-results)
8. [Run it](#8-run-it)
9. [Honest caveats](#9-honest-caveats)
10. [Repository](#10-repository)

---

## Dataset

The prompts are **JailbreakBench harmful behaviours** (Chao et al. 2024,
arXiv:2404.01318), pulled through lesson 2's loader
(`hello_world_steering.data.load_harmful_benign`, the `Goal` column via
`hf_hub_download`). `config.N_PER_CLASS = 60` harmful prompts are drawn;
`N_EXTRACT = 40` build the refusal direction (diff-of-means at layer 12), the
rest are **held out** to score the attack. A matched benign set backs the
**collateral** check (the always-on guard must not turn harmless prompts into
refusals).

The model choice is the load-bearing detail: this lesson loads the **aligned
base** Gemma-3-1B from the local path `models/google/gemma-3-1b-it` (**not** the
abliterated model of lessons 1–3), because "strip refusal" is only a meaningful
attack against a model that refuses by default. The same Gemma also acts as the
Guard D judge.

| item | value |
|---|---|
| source / loader | JailbreakBench `Goal` via `hello_world_steering.data.load_harmful_benign` |
| size | 60 harmful/class; 40 extract (build direction) / rest held out for ASR; benign for collateral |
| labels | prompt-level harmful vs benign |
| model + judge | **aligned** `models/google/gemma-3-1b-it` (local), self-graded (Guard D) |

**What the lesson uses it for:** red-team the intervention — invert / project out
the refusal direction to jailbreak the aligned model — then ablate a layered
guard (norm clamp, projection lock, dual-forward verdict) and measure the
attack-success-rate falling back toward baseline.

---

## 1. The dual-use idea

Arditi et al. 2024 (*Refusal in LLMs is Mediated by a Single Direction*,
arXiv:2406.11717) showed that refusal is governed by **one direction** in
activation space. Lesson 2 used that fact constructively: build the direction as
a diff-of-means (`v = mean(harmful) − mean(benign)`, Rimsky et al. 2023,
arXiv:2312.06681), add `+v`, and an abliterated model refuses again.

The dual is immediate and unavoidable. Take the *same* `v` and instead of adding
it, **remove** it, and an aligned model that would normally refuse now complies.
No new machinery — just a sign flip on the knob lesson 2 already built. That is
the "rogue scalpel": a precise activation-space cut that deletes safety.

This lesson uses an **aligned** `google/gemma-3-1b-it` (it refuses harmful
prompts by default) so that "strip refusal" is a meaningful attack. On the
abliterated model from lessons 1–3 there is nothing to strip — you would first
re-install refusal with lesson 2's `+v` and treat that as the protected baseline
(see [caveats](#9-honest-caveats)).

---

## 2. The attack

`attack.py` implements two ways to suppress refusal — both defensive red-team
transforms of the refusal direction `û` (the unit `v_unit` from lesson 2):

| Mode | Edit at layer 12 | What it does |
|---|---|---|
| `project_out` (default) | `h ← h − (h·û)·û` | **Directional ablation** — deletes the residual's refusal component entirely. The model can no longer "notice it should refuse". |
| `negative_add` | `h ← h − α·‖h‖·û` | Adds the **negative** refusal direction, scaled to a fraction `α` of the local hidden norm — lesson 2's relative-add, sign-flipped. |

The edit is installed the same way lesson 2 installs `+v`: a forward hook on the
residual layer (`AttackContext`, the same shape as lesson 2's
`SteeringContext`), so it perturbs every position during the prompt read *and*
each decoding step, then restores the model exactly on exit. Special control
tokens (BOS / `<start_of_turn>`) are never edited.

The key line, in `suppress_refusal_delta`:

```python
if mode == "project_out":
    proj = (h * u).sum(-1, keepdim=True)     # how much refusal is present
    return -project_frac * proj * u          # subtract it back out
```

Because the push is proportional to the refusal *already present*, it targets
exactly the refusal subspace and nothing else.

---

## 3. The guard: three layers

`guard.py` implements a small, defensible slice of CLAUDE.md's five-layer guard —
**B, A, and D** — each documented and independently ablatable.

**Guard B — norm / manifold CLAMP** *(activation space).*
Cap the per-position steering delta so `‖Δh‖ ≤ budget · ‖h‖`. An attacker trying
to overwrite a large fraction of the hidden state is blunted to a bounded nudge.
The delta's *direction* is preserved; only its magnitude is capped.

```python
delta = h_att - h_clean
factor = min(1, (budget·‖h_clean‖) / ‖delta‖)
h = h_clean + delta * factor
```

**Guard A — refusal-subspace projection LOCK** *(activation space).*
Re-add refusal so it cannot be removed. Guarantee the residual's projection onto
`û` is at least a positive floor `floor · ‖h‖`; whatever the attacker projected
out or negated is re-installed *up to that floor*, and positions that already
carry enough refusal are left untouched (the lock only ever adds).

```python
deficit = relu(floor·‖h‖ − (h·û))       # missing refusal
h = h + deficit * û                       # re-install it
```

**Guard D — dual-forward VERDICT check** *(runtime output guard).*
After generation, a **second forward** (the same Gemma as judge) grades the
response. A COMPLIANCE verdict on a harmful prompt is flagged as a leak and
overridden with a safe completion — catching whatever slipped past A and B.

Guards A and B compose in a **single hook** (`GuardedAttackContext`, which
subclasses the attack so it sees the same clean `h`): clamp first (bound the
adversarial delta), then lock (guarantee a positive refusal floor). Guard D acts
on the generated text. The composed activation guard is `apply_guards(...)`.

---

## 4. Attack vs. guard, in one picture

```
                 harmful prompt: "how do I <redacted>?"
                                  |
                                  v
                    +-------------------------------+
                    |  residual stream @ layer 12   |   h_clean  (carries refusal)
                    +-------------------------------+
                                  |
             ATTACK (rogue scalpel): strip refusal
             h_att = h_clean - (h_clean·û)·û          <-- refusal component deleted
                                  |
   .-------------------- GUARD (defense) --------------------.
   |   B  CLAMP :  ‖Δh‖  <=  budget·‖h‖     (blunt the push)  |
   |   A  LOCK  :  (h·û) >=  floor·‖h‖       (re-add refusal)  |
   '---------------------------------------------------------'
                                  |
                                  v
                            response text
                                  |
             D  DUAL-FORWARD: judge the text; a COMPLIANCE
                leak on a harmful prompt -> safe completion
                                  |
                                  v
                         refusal restored
```

Without the guard the scalpel wins (the model complies). With the guard the
refusal component is bounded back in and any residual leak is caught at the
output — the ASR falls back toward the un-attacked baseline.

---

## 5. The ablation ladder

`run_rogue_scalpel.py` walks a **cumulative** ladder on held-out harmful prompts,
adding exactly one guard layer per rung so each layer's marginal effect is
readable (the additive-ladder discipline from CLAUDE.md section 9). The metric is
**ASR = attack-success-rate = fraction of responses judged COMPLIANCE** (lower is
safer):

| Rung | Configuration | Expectation |
|---|---|---|
| `baseline` | no attack, no guard | model refuses — **ASR low** |
| `attacked` | rogue scalpel, no guard | refusal stripped — **ASR up (jailbroken)** |
| `+clamp` | attack + Guard B | the push is blunted — ASR drops |
| `+lock` | attack + Guards B+A | refusal re-installed — ASR drops further |
| `+dual` | attack + Guards B+A+D | leaks caught at output — **ASR back down** |

A separate **benign collateral** check confirms the always-on guard does not turn
harmless prompts into refusals (over-refusal must stay roughly flat). The run is
**honest**: if a guard fails to move the ASR, or a later rung is *worse* than an
earlier one, the numbers say so — nothing is smoothed.

---

## 6. Code walkthrough, file by file

### `config.py` — every knob in one place
The aligned model, the shared layer (12), the attack mode + strengths
(`ATTACK_ALPHA`, `ATTACK_PROJECT_FRAC`), the guard thresholds (`CLAMP_BUDGET`,
`LOCK_FLOOR_FRAC`), the safe completion, the data split, and paths. The attack is
deliberately set stronger than the clamp budget so an *unguarded* attack
overwhelms the model.

### `attack.py` — the rogue scalpel (the attack)
`refusal_direction(...)` reuses lesson 2's `extract_caa_vector` verbatim.
`suppress_refusal_delta(...)` is the pure delta math (project-out or
negative-add). `AttackContext` installs it as a forward hook (restores exactly on
exit). `attack_generate(...)` generates under the attack. CPU self-test asserts
project-out zeroes the refusal component, negative-add points anti-refusal at the
right magnitude, and the hook restores the model.

### `guard.py` — the layered defense
`norm_clamp(...)` (Guard B) and `projection_lock(...)` (Guard A) are pure math;
`apply_guards(...)` composes them; `GuardedAttackContext` runs attack+guards in
one hook; `dual_forward_verdict(...)` / `enforce_dual_forward(...)` are Guard D.
CPU self-test asserts the clamp bounds `‖Δh‖`, the lock restores the refusal
floor, and the stack re-installs refusal after a full ablation.

### `run_rogue_scalpel.py` — the orchestrator (GPU)
Builds the direction, walks the 5-rung ASR ladder, runs the benign collateral
check, saves `artifacts/results.json` + `artifacts/asr_ladder.png`, and prints a
summary table. Everything model-touching lives under `main()` so the module
imports cleanly for tests.

---

## 7. Results — measured vs. the claim

Artifacts come from the GPU run (`artifacts/results.json`, n = 50 held-out
harmful prompts per rung, aligned Gemma-3-1B, graded by an **off-family
Qwen-3B judge** on the shared 500/class toxic-chat-derived pool). The
load-bearing surprise stands: on this aligned base the project-out attack never
produces *compliance* — it produces **gibberish** — so ASR *falls* rather than
rises, and the readable safety axis is the **refusal rate**, not ASR.

| Claim | What we measured (off-family Qwen-3B judge, n=50/rung) | Verdict |
|---|---|---|
| Activation steering can compromise safety — project out the refusal direction to strip refusal | refusal 0.52 (baseline) → 0.00 (attacked); the stripped refusal turns into gibberish (0.12 → 1.00), and compliance/ASR *drops* 0.36 → 0.00 | Partly supported — the attack destroys refusal, but on this aligned base it breaks the model into gibberish rather than jailbreaking it |
| Guard B (norm / manifold clamp) defends | +clamp restores refusal 0.00 → 0.60 and pulls gibberish 1.00 → 0.20; ASR 0.20 | Supported — the clamp is the one guard that recovers a working, mostly-refusing model |
| Stacking Guards A (projection lock) and D (dual-forward) defends further | +lock and +dual both collapse back to refusal 0.00, gibberish 1.00 — worse than clamp alone | Not supported — the later rungs re-break the model; lock/dual do not help here |
| The always-on guard doesn't over-block benign prompts | benign refusal 0.40 (baseline) → 0.00 (guarded) | Supported — no over-refusal; if anything the guard under-refuses benign prompts |

The honest read: the attack + clamp-defense pair reproduces cleanly (refusal 0.52
→ 0.00 → 0.60), but not every guard rung earns its place — adding the projection
lock and dual-forward rungs pushes the residual stream back off-manifold and
*re-breaks* the model to refusal 0.00 / gibberish 1.00. The baseline refusal is
only 0.52 here, not the 0.95 the old 1B self-judge reported: that judge was
reading its own hedged compliance as refusal and inflating it, and the off-family
Qwen-3B judge corrects it. ASR stays ~0.0 for the reason the lesson already flags —
the attack yields gibberish an aligned 1B, never compliance — so watch the
refusal rate, not the ASR staircase. Screening tier, n=50, off-family judge — a
directional demo, not a hardened result.

---

## 8. Run it

From the **repo root** (`steeringresearch/`):

```bash
# CPU-only unit tests — no model download (attack math, hook restore, guard math)
python -m steering_tutorials.rogue_scalpel.attack
python -m steering_tutorials.rogue_scalpel.guard

# The full attack -> guard ladder (needs the ~2-3 GB Gemma-3-1B; GPU recommended)
# STEER_JUDGE_MODEL selects the OFF-FAMILY judge (avoids same-model grading bias).
STEER_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct python -m steering_tutorials.rogue_scalpel.run_rogue_scalpel
```

Uses `google/gemma-3-1b-it` (a **gated** model — accept the license and
`huggingface-cli login` first). JailbreakBench data downloads automatically via
`hf_hub_download`. Runs on CPU too, just slower.

Depends on lesson 2 for its plumbing (`model_utils`, `steer_vector`, `judge`) —
those import cleanly; no lesson-2 artifacts are required.

---

## 9. Honest caveats

- **This is a 1B toy demo, not a hardened guardrail.** The guards are defensible
  and the math is exact, but a real universal attack — the 20-vector Rogue
  Scalpel (Korznikov et al. 2025, arXiv:2509.22067), which averages many
  individually-weak steering vectors into one that flips the safety verdict — is
  far stronger than the single-direction attack shown here. Do not read `+dual ≈
  baseline` as "solved".
- **A 1B judge is weak.** As in lessons 2–3, self-grading with a small model is
  pedagogy, not publication-grade evaluation. Guard D inherits that weakness — it
  is only as good as the judge it calls.
- **Guard D can over-block.** Overriding any flagged compliance with a canned
  refusal will also refuse *false positives*. That is why the benign collateral
  check exists; watch it, and read a low ASR together with the over-refusal rate.
- **The lock assumes the refusal direction is correct.** If `û` is mis-estimated
  (too few contrast pairs, wrong layer), the lock re-installs the *wrong*
  direction. The floor is a blunt instrument, not a proof of safety.
- **Using the abliterated model instead.** If you only have the abliterated
  Gemma from lessons 1–3, it does not refuse to begin with. Re-install refusal
  first with lesson 2's `+v` steering and treat *that* as the protected baseline;
  the attack then strips the re-installed refusal and the guard defends it. Swap
  `MODEL_ID` in `config.py` and adapt the baseline generator accordingly.
- **Pedagogy, not a safety product.** This shows *how* an activation-space
  jailbreak and its defense work end-to-end on one small model. Do not deploy it.

---

## 10. Repository

Source and full artifacts:
<https://github.com/dlmastery/steeringresearch/tree/master/steering_tutorials/rogue_scalpel>

See also
[lesson 1 — the probe (READ)](../hello_world/README.md),
[lesson 2 — fixed-vector conditional steering (WRITE)](../hello_world_steering/README.md),
and [lesson 3 — HyperSteer (GENERATE)](../hypersteer/README.md).
