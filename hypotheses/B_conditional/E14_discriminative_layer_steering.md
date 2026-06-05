# E14 — Discriminative-Layer Steering: Capability Preservation

> **One-line claim:** Steering only at layers where class means have opposite
> sign (discriminative layers) preserves MMLU capability better than all-layer
> steering at fixed behavior efficacy.
>
> **Source design space:** Block B — Conditional / Gated Steering (E9–E16).
>
> **Implementation status:** `PENDING — UNTESTED`. Requires per-layer
> discriminative-layer identification and the full five-axis eval bundle.

---

## In Plain English

**What we're testing, simply:** We can inject the nudge at every processing step, or
only at the steps that actually carry the target behavior. This asks whether nudging
only the "relevant" steps keeps the model just as effective at the behavior while
hurting its general smarts less.

**Key terms (defined here so you don't have to look anything up):**
- **Language model (LLM):** an AI that predicts the next word; here, small Gemma
  models.
- **Steering:** nudging the model's behavior by adding a direction to its internal
  state while it writes.
- **Steering vector:** the specific direction of the nudge.
- **Residual stream:** the model's running internal state, where the nudge is added.
- **Layer:** one of the model's stacked processing steps.
- **alpha (strength):** how hard the nudge pushes.
- **All-layer steering:** injecting the nudge at every step at once — strong effect,
  but more damage to general ability.
- **Discriminative layer:** a step that genuinely encodes the behavior contrast (its
  "yes" and "no" examples point opposite ways). Nudging here does real work.
- **Neutral layer:** a step that doesn't encode the contrast; nudging here mostly just
  adds noise and hurts general ability without helping the behavior.
- **Selective steering:** injecting only at the discriminative layers.
- **Capability / MMLU:** whether the model stays smart, measured by a multiple-choice
  quiz (MMLU). We want a smaller drop here.
- **Behavior efficacy:** how well the nudge actually produces the target behavior.

**Why we're doing this (the point):** Steering always costs a little general
intelligence. If we can get the same behavior change while touching fewer steps, we
pay a smaller "intelligence tax" — a better trade-off for real use.

**What the result would mean:** A positive result means targeting only the relevant
steps preserves the model's smarts at no cost to the behavior. A negative result
means you can't separate the two — the damage comes with the effect.

See [`../GLOSSARY.md`](../GLOSSARY.md) for any other term.

---

## 1. Motivation (>= 100 words)

All-layer steering — injecting the behavior vector at every layer of the
residual stream simultaneously — is known to amplify behavior effects but also
amplifies capability degradation. The per-layer capability cost is not uniform:
some layers are "discriminative" for the target behavior (their class means
have opposite sign relative to the steering direction, meaning the layer
actively encodes the behavior contrast) while others are "neutral" (class means
are co-directional, meaning the injection fights against the layer's natural
representation). Selective Steering (arXiv:2601.19375) introduces the principle
of injecting only at discriminative layers — a gate on AXIS 1 (site) rather
than AXIS 5 (condition). The mechanism is that non-discriminative layers carry
general-purpose linguistic representations; injecting there adds noise to
capability-relevant computations without contributing to behavior change.
Restricting injection to discriminative layers should preserve these layers'
contributions to general intelligence (MMLU, ARC) while achieving the same
behavior efficacy as all-layer steering, because the behavior-relevant
information is concentrated in the discriminative subset.

---

## 2. Formal Hypothesis (>= 50 words)

Because the class means of harmful vs harmless activations are opposite-signed
at discriminative layers (by definition) and co-signed at neutral layers, a
behavior injection alpha*v at a neutral layer performs zero net behavior work
(the model's internal direction and the injected direction reinforce each other,
not contrast) while still displacing h in a direction that disrupts neutral-
layer computations (capability harm). Restricting injection to opposite-signed
layers maximizes behavior work per unit of off-shell displacement. Formal claim:
on Gemma-2-2B-it, Selective Steering at discriminative layers achieves >= 95%
of all-layer behavior efficacy while reducing MMLU drop by >= 30% (relative)
at the same nominal alpha, measured at 3-seed median.

---

## 3. Falsifier (>= 30 words)

If discriminative-layer steering achieves less than 95% of all-layer behavior
efficacy at matched alpha, OR if MMLU drop is not reduced by at least 30%
(relative) compared to all-layer steering, the hypothesis is FALSIFIED.
If discriminative layers exist but are a large fraction of all layers (> 60%),
the practical benefit of layer selection is diminished and status is NEAR-MISS.

---

## 4. Citations (Citation Rigor >= 80 words)

```
Goh, Gabriel, et al. (anon) 2025 arXiv 'Selective Steering: Discriminative
Layer Identification for Targeted Activation Steering' (arXiv:2601.19375) —
introduces discriminative-layer selection as a criterion for restricting
steering to layers where the class contrast is highest; the method is the
direct object of this hypothesis; their reported MMLU preservation result
[NEEDS VERIFICATION on Gemma-2-2B].

Arditi, Andy, et al. 2024 arXiv 'Refusal in Language Models Is Mediated by a
Single Direction' (arXiv:2406.11717) — identifies layers where refusal direction
is most active; the discriminative-layer criterion should align with these
high-refusal-activity layers.

Korznikov, A., et al. 2026 ICML 'The Rogue Scalpel: Activation Steering
Compromises LLM Safety' (arXiv:2509.22067) — Guard Layer C (avoid fragile
mid-layers) is mechanistically related: injecting at non-fragile layers reduces
rogue compliance; discriminative-layer selection may naturally avoid the fragile
layers identified by the paper (peak damage at early-middle layers).

Turner, Alexander Matt, et al. 2023 arXiv 'Activation Addition: Steering
Language Models Without Optimization' (arXiv:2308.10248) — all-layer ActAdd is
the baseline being compared; provides the all-layer protocol that discriminative-
layer selection is designed to improve upon.
```

---

## 5. Mechanism

### 5.1 Discriminative layer identification

For behavior B (harmful vs harmless):

    mean_harmful(l) = E[h_l | prompt harmful]
    mean_harmless(l) = E[h_l | prompt harmless]
    v_B(l) = mean_harmful(l) - mean_harmless(l)

Layer l is discriminative iff:

    sign(v_B(l)) ≠ sign(mean(h_l))   (opposite-signed: injection fights the
                                        layer's mean direction)

or equivalently:

    cos(v_B(l), mean_harmless(l)) < 0   (the steering direction points away
                                          from the harmless cluster)

The set of discriminative layers D_B is typically a subset of all layers.

### 5.2 Expected layer count

From Selective Steering (arXiv:2601.19375): discriminative layers tend to cluster
in specific depth ranges. For a 26-layer model, D_B may contain 6-10 layers.
Injecting only at these 6-10 layers should achieve comparable behavior efficacy
to all-layer injection because these layers carry the behavior-relevant
representational contrast.

### 5.3 Capability preservation mechanism

At non-discriminative layers, the behavior vector and the layer's natural mean
are co-directional. Injecting at these layers adds a large component in the
direction the layer is already moving — this is mechanistically equivalent to
amplifying the layer's output without adding behavioral contrast, which disturbs
the norm budget and decrements general-purpose computations without behavioral
benefit.

---

## 6. Predicted Delta

| Metric | Predicted value | Rationale |
|---|---|---|
| Behavior efficacy (discriminative-layer vs all-layer) | >= 95% | Core claim |
| MMLU drop reduction (discriminative vs all-layer) | >= 30% relative | Core claim |
| Fraction of layers in D_B | 25-40% | Selective Steering prior; 6-10/26 layers |
| PPL (discriminative vs all-layer) | [0, -0.5] lower | Fewer writes = lower PPL disruption |
| Off-shell displacement Delta||h|| | [20%, 40%] lower | Fewer injection points |

Pre-registered; [NEEDS VERIFICATION] on Gemma-2-2B.

---

## 7. Experimental Protocol

### 7.1 Primary experiment

- **Model:** Gemma-2-2B-it, 4-bit; smoke on Gemma-3-1B-it
- **Behavior:** refusal direction (same as E9)
- **Baseline:** all-layer steering at alpha in {10, 15, 20} (relative)
- **Discriminative-layer steering:** same alpha, injection only at D_B layers
- **Metrics:** behavior efficacy, MMLU delta, PPL, offshell Delta||h||
- **Layer count |D_B|:** report as a result
- **Seeds:** 3 (screening); 7 if MMLU preservation is within noise
- **Rogue Scalpel check:** JailbreakBench CR reported for both conditions

### 7.2 Where it shines

Most useful when the behavior is concentrated in a specific depth range
(e.g., late-middle layers for refusal). Will show the largest capability
preservation benefit when D_B is small (few discriminative layers) relative
to all layers.

---

## 8. Cross-References

- **E13** (early condition layer): condition check at early L_c; injection at
  discriminative L_b — the two constraints may need joint optimization
- **E16** (capability tax of gated vs always-on): this experiment isolates
  the within-unconditional (always-on) capability cost of layer selectivity
- **N5** (norm budget): fewer injection layers means lower cumulative ||delta h||
- **Rogue Scalpel Guard C** (corpus/steering-first-principles-v2): "avoid fragile
  mid-layers" — discriminative layers may naturally avoid this region
- **IDEA_TABLE.md** Block B row E14

---

## 9. Committee Q&A

**Q: Isn't this just a variant of layer selection (E2, which was FALSIFIED)?**

> E2 tested the Fisher ratio as a predictor of single-layer behavior efficacy
> (Spearman rho = 0.14, FALSIFIED for Gemma-270m). Discriminative-layer
> selection (E14) selects layers by the sign of the class-mean alignment,
> not the Fisher ratio magnitude. The two criteria can diverge. Moreover, E14
> selects a SUBSET of layers for injection (not a single layer), which is a
> different protocol. The E2 failure on Gemma-270m does not foreclose E14 on
> Gemma-2-2B.

**Q: Does the opposite-sign criterion have a mechanistic grounding?**

> Yes: at a discriminative layer, the model's internal representation actively
> contrasts the two classes. Injecting in the same direction as this contrast
> reinforces the natural discriminative computation. At a non-discriminative
> layer, the injection fights the natural flow — equivalent to adding noise
> in the capability computation's reference frame.

---

## 10. Verification Checklist

- [ ] D_B layers identified (opposite-sign criterion applied at each layer)
- [ ] |D_B| / 26 layers reported; distribution across depth noted
- [ ] Behavior efficacy: discriminative vs all-layer at 3 alpha values
- [ ] MMLU delta: discriminative vs all-layer reported per alpha
- [ ] Delta||h|| (offshell): discriminative vs all-layer compared
- [ ] JailbreakBench CR reported for both conditions
- [ ] IDEA_TABLE.md row E14 updated

---

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: UNTESTED / PENDING. Code needed:
  (a) per-layer class-mean computation; (b) discriminative-layer mask applied
  during injection; (c) full five-axis eval at Rung 2. Blocked on E9 pipeline
  and five-axis eval bundle.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-B.*

### Prior plausibility

MEDIUM-HIGH. The opposite-sign criterion is mechanistically motivated and
the Selective Steering paper reports positive results [NEEDS VERIFICATION].
The main uncertainty is whether D_B on Gemma-2-2B is small enough to provide
meaningful capability preservation (if D_B = all layers, the method collapses
to all-layer steering).

### Confounds

1. **Alpha calibration:** all-layer vs discriminative-layer steering at the
   SAME alpha is not iso-efficacy — discriminative-layer steering at the same
   alpha will have lower efficacy (fewer injection points). The comparison
   should be at iso-efficacy (find the alpha that gives equal behavior
   efficacy in each condition, then compare MMLU).

### Verdict

**NOVEL+TESTABLE** for Gemma-2-2B. The iso-efficacy confound is the most
important design choice — the protocol should be clarified before running.

---

## Pseudocode & Methodology

This hypothesis restricts the behavior write to **discriminative layers** (where the
class-mean sign opposes the steering direction) and asks whether that preserves MMLU at
matched behavior efficacy. The knob varied is the **set of injection layers** (D_B subset
vs all layers) — an AXIS-1 (site) gate, not an AXIS-5 (condition) gate.

### 1. Steering-vector recipe

```python
# Per-layer DiffMean behavior vector (refusal), used both to steer and to select layers.
for L in range(n_layers):
    H_L  = collect_activations(model, tok, harmful_vs_harmless, layer=L)
    v_B[L] = diffmean_vector(H_L.pos, H_L.neg)       # mean(harmful)-mean(harmless) @ L
    mean_harmless[L] = H_L.neg.mean(0)
# Discriminative iff the steer direction points AWAY from the harmless cluster:
D_B = [ L for L in range(n_layers) if cosine(v_B[L], mean_harmless[L]) < 0 ]
```

### 2. Experiment procedure

```text
1. Identify D_B (opposite-sign layers); report |D_B|/n_layers.
2. ALL-LAYER baseline:   inject alpha*v_B[L] at every L (apply_operation 'add').
3. DISCRIMINATIVE:       inject alpha*v_B[L] only for L in D_B.
4. Match on behavior efficacy (find iso-efficacy alpha per condition — see §confound), then:
5.     measure MMLU delta (eval.mcq_accuracy), PPL (eval.perplexity),
              offshell Delta||h|| (geometry.offshell_displacement), JailbreakBench CR.
6. for alpha in {10,15,20} (relative); seeds = 3 (screening).
```

### 3. Measurement & decision rule

PRIMARY metrics: behavior efficacy ratio (discriminative / all-layer) and MMLU-drop
reduction. FALSIFIER (§3): FALSIFIED if discriminative-layer steering achieves `< 95%`
of all-layer behavior efficacy at matched alpha, OR if MMLU drop is not reduced by
`>= 30%` (relative); NEAR-MISS if D_B is `> 60%` of all layers (little practical benefit).
Pre-registered (§6): behavior `>= 95%` of all-layer; MMLU-drop reduction `>= 30%`;
`|D_B|` = 25–40% of layers; PPL and Δ‖h‖ both lower for the discriminative subset.
Verdict: SUPPORTED iff efficacy is preserved AND MMLU tax drops ≥30%. The §9 caveat: the
comparison must be **iso-efficacy** (equal behavior, then compare MMLU), not iso-alpha.

### 4. Where the code is / status

Per-layer class-mean / DiffMean computation reuses `extract.collect_activations` +
`extract.diffmean_vector`; injection uses `hooks.apply_operation('add', ...)` over a
layer mask; metrics use `eval.py` + `geometry.py`. **New machinery**: the multi-layer
masked-injection dispatch (inject at a *subset* of layers in one forward pass) plus the
five-axis eval bundle wired for harmful/benign. Status **UNTESTED**, blocked on the E9
pipeline + eval bundle. (Distinct from the FALSIFIED E2: E14 selects a *subset* by
class-mean *sign*, not a single layer by Fisher *magnitude*.)

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/E14.md`.
