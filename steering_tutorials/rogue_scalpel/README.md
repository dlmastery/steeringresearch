# Rogue Scalpel — red-teaming a steering intervention, and defending it

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

## 7. Results

Artifacts come from the GPU run, which happens **after** this doc is written — so
the values below are placeholders until `results.json` populates them.

**`artifacts/asr_ladder.png`** — a bar per rung. The expected shape is a low
`baseline`, a tall red `attacked` peak (the jailbreak works), then a staircase
back down as `+clamp`, `+lock`, `+dual` each remove attack success.

| Rung | ASR (fraction complying) | Read as |
|---|---|---|
| baseline | (see `results.json`) | model's default refusal |
| attacked | (see `results.json`) | how well the scalpel jailbreaks it |
| +clamp | (see `results.json`) | marginal effect of the manifold clamp |
| +lock | (see `results.json`) | marginal effect of the projection lock |
| +dual | (see `results.json`) | residual leaks caught at the output |

The headline is the **delta**: `attacked` ASR minus `+dual` ASR is how much
attack success the guard removed. The guard "passes" this toy demo if `+dual`
returns ASR to roughly the `baseline` level **without** raising benign
over-refusal.

---

## 8. Run it

From the **repo root** (`steeringresearch/`):

```bash
# CPU-only unit tests — no model download (attack math, hook restore, guard math)
python -m steering_tutorials.rogue_scalpel.attack
python -m steering_tutorials.rogue_scalpel.guard

# The full attack -> guard ladder (needs the ~2-3 GB Gemma-3-1B; GPU recommended)
python -m steering_tutorials.rogue_scalpel.run_rogue_scalpel
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
  Scalpel (Korznikov et al. 2026, arXiv:2509.22067), which averages many
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
