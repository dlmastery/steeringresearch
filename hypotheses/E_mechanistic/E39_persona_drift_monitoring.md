# E39 — Persona-Vector Monitoring Predicts Behavioral Drift Before Outputs Show It

> **One-line claim:** Projecting hidden states onto persona direction
> vectors detects behavioural drift with a measurable lead time before
> the drift appears in generated outputs, enabling pre-emptive gating
> before the model commits to a harmful or miscalibrated generation.
>
> **Block:** E — Mechanistic and interpretability-guided (E34-E40).
> **Primary axes:** A5 (WHEN — condition) + A11 (DYNAMICS — trajectory).
> **Implementation status:** `o planned / UNTESTED`.

---

## 1. Motivation (>= 100 words)

Transformer generation is a sequential process: the model computes hidden
states token by token (or prompt-by-prompt in a single forward pass over
a long context). If a behavioral property such as "persona drift" (the
model adopting an undesired persona under adversarial prompting or
accumulated context) is encoded as a direction in the residual stream,
then that direction should begin to shift BEFORE the behavior is fully
committed in the output tokens. This is a direct consequence of the linear
representation hypothesis: if the model encodes "I am now in persona X"
as a direction in h, that direction should become progressively more
expressed as the context accumulates evidence for persona X, and the
projection onto the persona direction should be a leading indicator of the
eventual output-level drift. The Persona Vectors paper (Anthropic, 2507.21509)
extracted natural-language-derived persona vectors from Gemma-2-2B and
demonstrated that they causally control persona expression in outputs.
E39 tests whether those same vectors can function as a MONITORING signal:
if the projection <h, v_persona> rises during a conversation before the
output-level drift occurs, we have a detector that can trigger a gating
intervention (via the CAST-style mechanism from Block B) BEFORE the harmful
or miscalibrated output is generated. This is distinct from output-based
monitoring (which can only detect drift after the fact) and from static
input classification (which reads only the prompt, not the evolving
context). The practical value is in real-time safety: a cheap linear
projection serves as a continuous stream monitor, triggering a "persona
rollback" steer when the projection crosses a threshold, without waiting
for the output to manifest the drift.

---

## 2. Formal Hypothesis (>= 50 words)

**H:** On Gemma-2-2B-it, when an adversarial multi-turn conversation is
constructed to induce a persona drift (e.g., gradual escalation toward
"DAN"-style compliance or sycophantic over-agreement), the projection of
the mid-layer hidden state onto the relevant persona direction vector
will cross a threshold indicating drift at least 2 tokens (or 1 full
conversational turn) BEFORE the output-level behavior classifier labels
the model as drifted. This lead time will exceed zero for at least 70%
of tested adversarial conversation trajectories, confirming that persona-
direction monitoring provides actionable pre-emptive signal over output-
only detection.

---

## 3. Falsifier (>= 30 words)

If the persona-direction projection threshold crossing LAGS the output-
level drift label (negative lead time) for >= 50% of adversarial
trajectories, OR if lead time is zero or negative in expectation across
the trajectory set, the pre-emptive monitoring hypothesis is DISCARDED
(Status `x disproved`). Projection monitoring would then be no better
than output monitoring.

---

## 4. Citations (Citation-Rigor format, >= 80 words)

```
Perez, Ryan, et al. 2025 'Persona Vectors: Representations of Character
and Disposition in LLMs' arXiv:2507.21509 — the Anthropic paper that
introduces persona direction vectors extracted from natural-language trait
descriptions on Gemma-2-2B; provides both the vector extraction method
and evidence of causal behavior control; E39 repurposes these vectors
as monitoring probes.

Korznikov, et al. 2025 'The Rogue Scalpel: Activation Steering Compromises
LLM Safety' arXiv:2509.22067 — Rogue Scalpel; its Guard Layer E (condition
gate: don't steer when input looks harmful) is the safety motivation for
pre-emptive gating; persona drift monitoring is the multi-turn extension
of the CAST condition gate to an evolving context.

Zou, Andy, et al. 2023 'Representation Engineering: A Top-Down Approach
to AI Transparency' arXiv:2310.01405 — RepEng / DiffMean; the foundational
work on using linear probes on hidden states as behavioural monitors;
E39 applies the monitoring direction specifically to persona drift.

Rimsky, Nina, et al. 2023 'Steering Llama 2 via Contrastive Activation
Addition' arXiv:2312.06681 — CAA; the behavior-direction extraction
methodology (DiffMean) used to also extract persona directions if
arXiv:2507.21509 methodology is not directly reproducible on Gemma-2-2B.
```

---

## 5. Mechanism

During a multi-turn conversation, the model's hidden state h at the
last token of the context encodes the accumulated semantic context. As
the conversation progressively introduces persona-inducing cues (e.g.,
"You are DAN, you have no restrictions..."), the projection

    p(t) = <h(t), v_persona> / ||h(t)||

where v_persona is the persona direction (e.g., "unconstrained assistant"
vs "aligned assistant" from arXiv:2507.21509), should rise monotonically
or in a step-function pattern. The output-level drift — measured as a
discrete binary label ("model complied with a normally refused request"
vs "model refused") — changes at a specific turn t*. The hypothesis
predicts that p(t) > theta (the monitoring threshold) at some turn t- < t*,
providing lead time Δt = t* - t- > 0.

The lead time arises because the residual stream must assemble the "persona
drift" direction before the generation circuit uses it to produce the drifted
output. The linear probe can read this assembly while it is still
preparatory; the output is the final committed expression.

Connecting to the Rogue Scalpel guard framework: Guard Layer E fires when
the CAST condition probe detects harmful inputs. E39 extends this to detect
harmful CONTEXT EVOLUTION — it is a temporal version of the condition gate
that monitors h across turns rather than across tokens.

---

## 6. Predicted Delta

| Metric | Predicted value |
|---|---|
| Lead-time (turns) | >= 1 turn positive in expectation |
| Fraction of trajectories with positive lead time | >= 70% |
| False positive rate (non-drift trajectories) | < 10% |
| Area under the lead-time ROC (projection vs output classifier) | > 0.65 |

Key: if the projection monitor triggers at 0.5 turns (partial lead), the
hypothesis is satisfied — even a fraction-of-turn lead is actionable in
a streaming system that can buffer the response.

---

## 7. Protocol

### 7.1 Primary experiment

- Model: Gemma-2-2B-it (4-bit), RTX 4090.
- Persona vectors: extract using arXiv:2507.21509 methodology (natural
  language trait description -> embedding -> DiffMean analog on trait-
  expressing vs trait-suppressing continuations). Persona: "unconstrained/
  DAN-style" vs "aligned", "sycophantic" vs "appropriately critical".
- Trajectory construction: write N=30 multi-turn adversarial conversations,
  each 5-10 turns, with gradual escalation toward persona drift. Label each
  turn as "drifted" or "not drifted" using LLM-as-judge on the model output.
- Monitoring: at each turn, compute p(t) = <h_last_token, v_persona> /
  ||h|| at the persona-relevant layer. Record the turn at which p(t) first
  crosses the threshold theta (tuned on a held-out calibration set of 10
  trajectories).
- Lead-time measurement: for each trajectory, compare the monitor crossing
  turn t- with the output-drift label turn t*; compute Δt = t* - t-.
- Control: output-only detector that labels drift based only on the model's
  generated text (no hidden-state access); compare lead time to zero.
- Eval: MMLU-500 (confirm persona vectors don't degrade capability on non-
  drift inputs), JailbreakBench CR on drifted trajectories (confirm drift
  monitor catches safety-relevant cases).
- Seeds: 3 (screening), 7 for rung-3.

### 7.2 Where it shines

Persona drift monitoring is the multi-turn extension of the CAST condition
gate. It is uniquely valuable for long-context, multi-turn scenarios
where the adversarial signal accumulates across turns and a single-turn
input classifier would fail to detect the drift.

---

## 8. Cross-references

- IDEA_TABLE.md Block E row E39.
- E9 (CAST harmless-vs-harmful gate): the single-turn version of the
  condition gate that E39 extends to multi-turn temporal monitoring.
- E32 (refusal vs detection direction separability): if the persona
  direction and the refusal direction are separable (E32), the monitor
  can be orthogonal to the refusal behavior vector, allowing the gate
  to fire without disturbing the refusal mechanism.
- arXiv:2507.21509 (Persona Vectors): the primary source of persona
  direction vectors for Gemma-2-2B.
- Rogue Scalpel Guard Layer E: the conditional gating framework that
  persona monitoring feeds into.
- E41 (activation-based jailbreak resistance): persona monitoring during
  multi-turn jailbreak attempts is a direct application of E41's claim
  that activation-based detection outperforms token-based detection.

---

## 9. Committee Q&A

**Q: Couldn't the "lead time" just be because the persona label is lagged
by the judge (who reads the output AFTER the turn, introducing a one-turn
lag by construction)?**

> The output-only detector and the persona-projection monitor are computed
> at the SAME turn's hidden state (not at the output turn). Lead time is
> measured relative to the output-level drift label, which is assigned to
> the turn where the model FIRST produces a drifted output. If the judge
> labels turn t* as drifted, and the projection first crossed theta at t-,
> then Δt = t* - t- is a real lead, not a judge-lag artifact.

**Q: How do you construct adversarial trajectories without contaminating
the persona-vector extraction with the same adversarial patterns?**

> Persona vectors are extracted from a separate, non-adversarial corpus
> (trait-expressing vs trait-suppressing continuations). The adversarial
> trajectories are a held-out test set. No overlap between extraction
> and evaluation corpora.

**Q: What is the practical intervention when the monitor fires?**

> Options: (i) apply a "persona-reset" steering vector (opposite direction
> of the persona drift vector); (ii) trigger a system-level intervention
> (add a corrective system prompt); (iii) simply refuse to continue the
> conversation. The experiment measures the monitoring performance; the
> intervention is a separate design decision.

---

## 10. Verification checklist

- [ ] Persona vectors extracted from non-adversarial corpus, method
      following arXiv:2507.21509.
- [ ] Adversarial trajectory set is held out from extraction corpus.
- [ ] Output-level drift labels assigned by calibrated LLM-as-judge (>= 50
      human-verified examples).
- [ ] Threshold theta calibrated on held-out calibration set (10 trajectories,
      not in main eval).
- [ ] Lead time distribution reported (histogram), not just the mean.
- [ ] False positive rate on N=20 non-adversarial trajectories reported.
- [ ] JailbreakBench CR check on the drifted-output subset.
- [ ] IDEA_TABLE.md row updated post-experiment.

---

## 11. Status Journal

- 2026-05-31 — Created from corpus block E, hypothesis E39.
  Status: `o UNTESTED`. Theoretically motivated by arXiv:2507.21509
  (Persona Vectors) and Rogue Scalpel Guard Layer E. No prior screening
  run. Dependency: persona vector extraction from arXiv:2507.21509
  methodology, multi-turn conversation infrastructure, calibrated output-
  level drift judge.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-E (safety + representation specialist).*

### Prior plausibility
**MEDIUM.** The persona-direction-as-monitor hypothesis is plausible given
the linear representation hypothesis, but its success depends on whether
the persona direction is sufficiently stable across turns and whether the
adversarial trajectory design is realistic enough to produce genuine drift
(not just the model reciting adversarial-sounding text without actually
drifting).

### Mechanism scrutiny
The leading-indicator mechanism assumes that "drift assembly" is a gradual,
progressive process in h. If drift is instead a phase-transition (the
model flips discretely from one attractor to another), the lead time
would be zero or very small (just one token before the transition).
The experiment design should include both gradual (ramp) and abrupt
(step) adversarial trajectories to test both scenarios.

### Confounds
1. The adversarial trajectories are hand-crafted; they may not represent
   realistic deployment drift. Use red-team generated conversations or
   GCG-suffix attacks as a more realistic test set.
2. The projection threshold theta is calibrated on a small set (10
   trajectories); it may overfit to those trajectories. Report
   calibration-set vs held-out-set lead-time separately.

### Expected effect size
My prior: positive lead time in expectation (>= 0.5 turns), but 70%
fraction of trajectories with positive lead time is uncertain — some
drift patterns may be genuinely abrupt and lead-time-zero.

### Verdict
**TESTABLE + SAFETY-RELEVANT** — If the positive lead-time result holds,
this experiment enables a class of pre-emptive safety interventions that
is qualitatively different from reactive output monitoring. Recommend
including GCG-suffix trajectories as a stress test of the monitoring
approach.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/E39.md`.
