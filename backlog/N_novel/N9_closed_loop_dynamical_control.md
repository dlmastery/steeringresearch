# N9 — Steering as Control of a Latent Dynamical System

> **One-line claim:** Token-by-token generation is a discrete dynamical system in
> activation space; a proportional feedback controller (adjust alpha from current
> projection error onto the behavior direction) beats fixed-alpha on long generations
> and resists behavioral drift over 512 tokens.
>
> **Primary axes:** A3 (how-much/coefficient), A11 (dynamics/trajectory)
> **Status:** UNTESTED

---

## In Plain English

**What we're testing, simply:** Normally we pick one steering strength at the start
and keep it fixed for the whole answer — like setting a thermostat once and never
touching it. Over a long answer, the model can drift off-behavior. This doc borrows
the *thermostat* idea: keep watching how "on-behavior" the text currently is, and
automatically push harder when it drifts away and ease off when it's on target.

**Key terms (defined here):**
- **Steering / steering vector** — changing behavior by adding a chosen direction to
  the model's internal "thought" as it writes, instead of retraining.
- **Residual stream** — the model's running internal thought; what we read and edit.
- **alpha / strength** — how hard we push.
- **Open-loop (fixed alpha)** — set the strength once, never adjust it.
- **Closed-loop / feedback control** — keep measuring the behavior and adjust the
  strength on the fly (the thermostat).
- **Drift** — the answer wandering away from the target behavior as it gets longer.
- **Coherence** — whether the text stays fluent. Measured by **perplexity**.
- **Off-shell displacement** — how far a nudge knocks the thought off its healthy size
  (we check feedback doesn't make this worse).

**Why we're doing this (the point):** A self-adjusting steer should hold the behavior
steady across long answers without anyone hand-tuning the strength — and without
overshooting into broken text.

**What the result would mean:** A win means the thermostat-style steer drifts less and
stays readable over long generations. A loss means fixed strength was just as good (or
feedback made things worse).

See [`../GLOSSARY.md`](../GLOSSARY.md) for any other term.

---

## 1. Motivation (>= 100 words)

The standard activation steering paradigm applies a fixed perturbation alpha*v at
each layer of each forward pass throughout generation. This is open-loop control:
the magnitude alpha is set at the beginning and never updated in response to the
actual behavioral state of the generation. In control engineering, open-loop
control is fragile: external disturbances (new context tokens, topic shifts,
multi-turn dialogue) cause the system state to drift from the target, and the
fixed-alpha perturbation may become too strong (if the state drifts away from the
target) or too weak (if it drifts toward but then overshoots). The generating model
is a discrete dynamical system: at each step, the next token changes the KV cache
and activations, which is a state update. The behavior projection b(h) = cos(h, v)
is a measurable output of this system. A proportional feedback controller uses this
measurement to adjust the steering magnitude: alpha(t) = K_p * (b_target - b(h(t))),
where K_p is the proportional gain and b_target is the desired behavior level. This
closed-loop formulation is well-studied in control theory (PID control, Kalman filter)
and has the fundamental advantage of disturbance rejection: if the generation drifts
off-behavior, the controller automatically increases alpha; if it overshoots, alpha
decreases. Screening result S-2 shows that fixed-alpha steering on long generations
can degrade coherence substantially; N9 predicts that closed-loop control prevents
this degradation by responding to the actual generation state.

## 2. Formal Hypothesis (>= 50 words)

Let b(h_t) = cos(h_t, v) be the behavior projection at generation step t. Let
alpha_t = K_p * max(0, b_target - b(h_t)) be the proportional controller gain.
The claim is: over T=512 generated tokens, on Gemma-3-1B-it:

  (A) mean behavioral drift |b(h_T) - b_target| is at least 30% lower with the
      P-controller than with fixed alpha calibrated to match b(h_0);
  (B) log-PPL at T=512 is at most 15% higher with P-controller than with the
      same behavior achieved at T=1 (coherence maintained over long context);
  (C) P-controller does not increase peak off-shell displacement vs fixed alpha.

## 3. Falsifier (>= 30 words)

If P-controller behavioral drift is worse than fixed-alpha drift (>= 30% higher) at
any K_p value tested, or if P-controller PPL at T=512 is > 25% above T=1 baseline,
the closed-loop control hypothesis is FALSIFIED. Status moves to `FALSIFIED`.
If drift reduction is 10-29%, status is `INCONCLUSIVE`.

## 4. Citations (Citation Rigor >= 80 words)

```
GeoSteer 2026. arXiv:2601.10229 (Kazama et al.). GeoSteer steers CoT reasoning
by following latent manifold gradients across steps — a trajectory-aware method.
N9 is the complementary closed-loop version: instead of following a planned gradient
trajectory (open-loop with manifold knowledge), N9 reacts to the actual state.
GeoSteer demonstrates that multi-step trajectory awareness improves steering by
+0.9 acc / +4.5 reasoning points [NEEDS REPLICATION]; N9 predicts similar gains
from feedback without requiring manifold gradient computation.

Turner et al. 2023. 'Activation Addition' arXiv:2312.06681. Fixed-alpha baseline;
N9 tests whether adaptive alpha outperforms this on long generations specifically.

Wurgaft et al. 2026. 'Manifold Steering' arXiv:2605.05115. The M_h <-> M_y link
defines the "target manifold" that the P-controller maintains the activation near.
Drift in M_y corresponds to drift in b(h_t); the P-controller is the feedback
implementation of manifold maintenance.

Raval et al. 2026. 'Curveball Steering' arXiv:2603.09313. Curveball optimizes
the steering PATH; N9 optimizes the steering MAGNITUDE along a fixed-direction path.
The two approaches are orthogonal: combining curved path (N13) with feedback
magnitude (N9) is the full trajectory-aware closed-loop controller.
```

## 5. Mechanism

At generation step t, the model produces token t and updates its KV cache. The
new context changes the residual stream at the current step: h_t = f(h_{t-1}, k_t)
where k_t is the new key-value pair. The behavior projection b(h_t) = cos(h_t, v)
measures how aligned the current activation is with the target behavior direction.

P-controller: alpha_t = K_p * (b_target - b(h_t))

When b(h_t) < b_target (drifting off-behavior), alpha_t > 0 (increase steering).
When b(h_t) > b_target (over-steered), alpha_t < 0 (reduce or reverse steering,
preventing over-refusal). The controller is stateless: it only uses the current
measurement b(h_t), not the history. This is the minimal form (proportional only);
the PI (proportional-integral) extension adds a memory term and is reserved for
follow-up.

K_p selection: K_p = 0.5 * alpha_calibrated / (b_target - b_initial) ensures that
the controller starts at approximately the calibrated fixed-alpha and adjusts from
there. For long generations, K_p can be tuned by searching over {0.25, 0.5, 1.0, 2.0}
in terms of the calibrated alpha.

Computational overhead: b(h_t) = cos(h_t, v) is one dot product per token per layer —
negligible compared to the attention computation. The only additional cost is the
subtraction and scaling of v at each step, which is already done in fixed-alpha steering.
The closed-loop controller is thus computationally free relative to fixed-alpha.

## 6. Predicted Delta

| Metric | Predicted Value | Rationale |
|---|---|---|
| Behavioral drift reduction at T=512 | 30% - 60% relative | P-controller's disturbance rejection |
| PPL at T=512 (P-controller vs fixed) | <= +15% above T=1 baseline | Coherence maintained by adaptive alpha |
| Peak off-shell displacement | <= fixed-alpha peak | Controller clips alpha when off-shell |
| K_p sensitivity | Moderate (works for K_p in [0.3, 2.0]) | P-control is robustly stabilizing |
| Improvement for longer generations | Increasing with T | Drift accumulates; feedback helps more |

## 7. Protocol

### 7.1 Primary experiment

- Model: Gemma-3-1B-it @L16 (established optimal layer)
- Behavior: 3 behaviors (safety/refusal, politeness, factuality)
- Generation: 512 tokens from 10 different seeded prompts
- Conditions: (a) fixed-alpha (alpha calibrated to match b(h_0) = b_target at t=0),
              (b) P-controller with K_p in {0.25, 0.5, 1.0, 2.0},
              (c) no steering (drift measurement baseline)
- Metrics: b(h_t) trajectory over t=0..512; final drift |b(h_512) - b_target|;
           log-PPL on the generated text; off-shell max(delta-||h_t||/||h_t||)
- b_target: set to mean b(h_0) across the 10 prompts at fixed alpha
- Seeds: 3 controller seeds x 10 prompt seeds = 30 trajectories per condition
- Wall-clock: ~4 hours on RTX 4090

### 7.2 Where it shines

Multi-turn dialogue (T >> 512): the drift problem compounds with dialogue length.
A P-controller enables consistent long-context steering (safety guardrails for
extended conversations) without recalibrating alpha after each turn. This is the
primary deployment motivation.

## 8. Cross-References

- N7 (parallel transport): transport-aligned vectors at each step provide the right
  v for each layer; N9 adjusts the magnitude; combining gives full trajectory control
- N11 (curvature-aware alpha): N11 sets the per-prompt alpha at the start;
  N9 adjusts it over the generation; they are complementary timescales
- N19 (trajectory beats endpoint): N19 distributes budget across layers;
  N9 distributes budget across time steps; both are "trajectory" vs "endpoint" ideas
- E22 (norm budget and collapse): the P-controller's alpha adjustment naturally
  enforces the norm budget: alpha drops when off-shell increases
- E48 (prefill vs per-token steering): E48 tests whether per-token recomputation
  matters; N9 predicts it DOES for long generations due to drift
- IDEA_TABLE.md: N9 row, axes A3+A11

## 9. Committee Q&A

**Q: Computing b(h_t) = cos(h_t, v) at each token requires extracting the hidden state
at each step. This breaks many inference optimizations (KV cache prefill). How
is this handled?**

> b(h_t) is computed from the hidden state that is already materialized during the
> forward pass for the alpha*v addition. The cost is one additional dot product per
> token, which is O(d_model) and < 0.1% of total compute. KV cache is not affected
> since the steering addition is post-attention.

**Q: Doesn't varying alpha over the generation produce incoherent text (non-stationary
steering signal)?**

> The P-controller varies alpha slowly and proportionally; it does not flip the
> sign of the steering unless the model is substantially over-steered. In practice,
> |alpha_t - alpha_{t-1}| is small for well-calibrated K_p, making the steering
> effectively stationary at short timescales. The non-stationarity is the POINT:
> it corrects for the accumulating drift in the generation.

**Q: How do you choose b_target? Setting it to the "ideal behavior level" is
a free parameter.**

> b_target is set to the behavior cosine observed at t=0 with the calibrated
> fixed-alpha, which is a behavior-specific constant determined before generation.
> The comparison is fair: fixed-alpha is calibrated to the same b_target at t=0,
> and the P-controller maintains b(h_t) near b_target throughout the generation.

## 10. Verification Checklist

- [ ] b(h_t) trajectory measurement implemented: extract h at injection layer per token
- [ ] Fixed-alpha baseline calibrated at b_target for each behavior
- [ ] P-controller with 4 K_p values implemented and verified stable (no oscillation)
- [ ] 512-token generation with behavioral drift measurement (30 trajectories per condition)
- [ ] PPL on generated text (WikiText continuation or behavior-relevant prompt completion)
- [ ] Off-shell trajectory measured; P-controller does not exceed fixed-alpha peak
- [ ] K_p sensitivity analysis: at what K_p does oscillation appear?
- [ ] Result in EXPERIMENT_LEDGER.md, IDEA_TABLE.md N9 row updated

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: UNTESTED. The P-controller idea is novel
  in the steering context. S-2 through S-8 document behavioral drift and coherence
  collapse at large alpha on short generations; N9 targets the long-generation regime
  which has not been tested. The E48 experiment (prefill vs per-token) is a direct
  prerequisite to check that per-step recomputation matters.

---

## Addendum: Research-Scientist Critique

### Prior plausibility

MEDIUM. P-control is a fundamental engineering concept with well-understood stability
properties. The application to LLM steering is novel but the mechanism is clear.
The main uncertainty is whether behavioral drift (b(h_t) decreasing over 512 tokens)
is a real and significant phenomenon — if drift is small in practice, the controller
provides little gain.

### Mechanism scrutiny

The b(h_t) measurement requires that the behavior direction v is stable across
the 512-token generation (i.e., the same v measured at position 0 is the right v
at position 512). If the residual stream's representation of the behavior concept
shifts over the generation (which is plausible for long documents), then the
fixed v is no longer the right target and b(h_t) becomes an unreliable measurement.
This is the "v drift" problem and is a genuine confound.

### Confounds

1. Token-level noise: cos(h_t, v) is noisy at the token level (it varies with the
   syntactic position of the token, not just behavioral state). Low-pass filtering
   b(h_t) over a window of 10-20 tokens before using it for control may be necessary.
2. The optimal K_p depends on the generation's dynamics (fast or slow drift).
   An unsuitable K_p causes oscillation rather than convergence.
3. If the generation is coherent and on-behavior by default (high b(h_t) naturally),
   the P-controller's alpha approaches zero and becomes inactive — no measurable
   advantage over no-steering.

### Does the closed-loop formulation specifically matter?

MODERATELY. An adaptive-alpha scheme that simply decays alpha over time (without
measuring b(h_t)) would also reduce long-generation drift. The P-controller
specifically matters over decay if the generation has non-monotonic drift — sometimes
off-behavior, sometimes on-behavior. The experiment should include a fixed-decay
baseline to isolate the feedback advantage.

### Literature precedent

Inference-time compute scaling (Wei et al. 2022, chain-of-thought) adapts computation
per step but not the steering signal. Speculative decoding and RLHF inference-time
algorithms adapt token selection; N9 adapts the steering vector magnitude. There is
no direct precedent for feedback control of activation vectors.

### Skeptical effect-size estimate

Behavioral drift reduction: 15-35% (vs claimed 30-60%). Rationale: if drift is
primarily driven by context-length effects on the attention mechanism (which are
layer-specific), a single-layer P-controller may not address the root cause. The
PPL preservation claim (<=15% above T=1) may hold if K_p is chosen conservatively.

### Minimum distinguishing experiment

Drift measurement alone: run fixed-alpha and no-steering over 512 tokens for one
behavior, 10 prompts. Measure b(h_t) trajectory. If drift is < 10% over 512 tokens,
the P-controller provides negligible value and N9 is not worth pursuing. Cost ~30 min.

### Verdict

TESTABLE-MEDIUM. The drift pre-check (30 min) is essential before committing to
the 4-hour full protocol. If behavioral drift over 512 tokens is large (> 20%),
the P-controller is motivated; if small, N9 is a solution to a non-problem.

---

## Pseudocode & Methodology

This section specializes [`../METHODOLOGY.md`](../METHODOLOGY.md). N9 makes alpha a proportional-feedback controller over the generation, correcting behavioral drift. **UNTESTED** — needs a per-token closed-loop steering hook.

### 1. Steering-vector recipe (P-controller over time)

The vector `v` is ordinary DiffMean; the novelty is a time-varying, state-dependent alpha:

```python
v = bank[L]["diffmean"]
# at each generated token t, read the behavior projection and adjust alpha:
b_t       = cos(h_t, v)                                  # one dot product per token, per layer
alpha_t   = K_p * max(0, b_target - b_t)                 # proportional feedback
#   h_t' = h_t + alpha_t * v        (METHODOLOGY §2 add, but alpha recomputed each step)
# geometry logged each step: offshell_displacement(h_t, h_t') must not exceed fixed-alpha peak
```

### 2. Experiment procedure

```text
1. Calibrate fixed-alpha so b(h_0) = b_target for each of 3 behaviors.
2. Generate 512 tokens under: (a) fixed-alpha, (b) P-controller K_p in {0.25,0.5,1.0,2.0},
   (c) no-steering drift baseline, (d) fixed-DECAY control (isolates feedback vs mere decay).
3. Record b(h_t) trajectory, final drift |b(h_512)-b_target|, log-PPL, peak off-shell.
4. 30 trajectories per condition (3 controller x 10 prompt seeds).
```

### 3. Measurement & decision rule

- **Primary metric:** behavioral-drift reduction at T=512, P-controller vs fixed-alpha.
- **Pre-registered falsifier (§3):** P-controller drift ≥ fixed-alpha drift at any K_p, OR PPL@512 > 25% above T=1 ⇒ FALSIFIED; 10–29% drift reduction ⇒ INCONCLUSIVE.
- **Verdict logic:** must beat the fixed-decay control to credit *feedback* (not decay) for the gain.

### 4. Where the code is / status

UNTESTED. Static-alpha injection (`SteeringContext`) and `geometry.offshell_displacement` exist, but the **per-token closed-loop hook** (read `cos(h_t,v)` and recompute alpha inside the decode loop, KV-cache-compatible) is the missing machinery — that is why N9 is UNTESTED.

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/N9.md`.
