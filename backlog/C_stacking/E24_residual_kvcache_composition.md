# E24 — Residual + KV-Cache Steering: Disjoint-Site Composition

> **One-line claim:** Residual + KV-cache steering (disjoint sites) compose
> with >= 85% of each effect on short generations but degrade on long ones
> from KV contamination.
>
> **Source design space:** Block C — Stacking and Multi-Vector Composition (E17–E26).
>
> **Implementation status:** `PENDING — UNTESTED`. Requires KV-cache steering
> implementation (separate from residual hook) and generation-length sweep.

---

## In Plain English

**What we're testing, simply:** We can nudge the model in two different *places*
inside it at once: in its running "thoughts," and in its short-term memory of the
words so far. The idea is that two nudges in different places shouldn't fight. We
also check a catch: on long passages, the memory nudge may slowly "leak" and
muddy the text.

**Key terms (defined here):**
- **Steering / steering vector:** nudging the model by adding a direction to its
  internal "thoughts"; the direction is the steering vector.
- **Residual stream:** the model's running internal "thoughts" as it writes — one
  place we can nudge.
- **KV-cache:** the model's short-term memory of the words it has already seen,
  which it reuses to write the next word — a *second*, separate place we can nudge.
- **Layer:** the processing step where we make the edit.
- **Alpha / strength:** how hard we push.
- **Coherence:** whether the text stays fluent and sensible.
- **Stacking:** using several nudges at once. **Disjoint sites:** the two nudges
  act in *different* places (thoughts vs memory), so they shouldn't collide.
- **Interference / contamination:** the nudges getting in each other's way; here,
  on long text the memory nudge can build up and gradually spoil the output.

**Why we're doing this (the point):** Nudging in two separate places is a natural
way to get two effects without them clashing. But we need to know how long it
lasts before the memory side starts to leak.

**What the result would mean:** A win means we can combine these two nudge sites
for short replies and get both effects nearly in full. The expected catch — worse
results on long text — tells us where the safe limit is.

See [`../GLOSSARY.md`](../GLOSSARY.md) for any other term.

---

## 1. Motivation (>= 100 words)

The intervention-site taxonomy (corpus: steering-stackable-vs-competing-analysis.md
§1.1) identifies residual-stream editing (CAA, DiffMean) and KV-cache steering
as operating on disjoint intervention sites — one modifies h during the forward
pass, the other edits the cached key-value pairs that carry context across tokens.
By the AXIS 1 (site) stacking rule, disjoint-site methods should compose cleanly
in principle. KV-cache steering (arXiv:2507.08799 [NEEDS VERIFICATION]) can inject
behavioral context by modifying the cached representations of earlier tokens,
while residual-stream CAA edits the current token's hidden state. Together, they
provide both a per-token behavioral push (residual) and a contextual behavioral
frame (KV-cache). The important qualification is that residual-stream steering
on later tokens contaminates the KV cache: the steered hidden state h is used
to compute the next-step key and value, propagating the edit into the context
that all subsequent tokens attend to. Over long generations, this contamination
accumulates. SKOP (arXiv:2605.06342) exists precisely to mitigate this: key-
orthogonal projection prevents the residual edit from leaking into the keys.
This experiment tests the generation-length degradation and the SKOP mitigation.

---

## 2. Formal Hypothesis (>= 50 words)

Because the residual-stream edit and the KV-cache edit operate on disjoint
mechanisms at token 0 (the residual edit modifies h at the current forward pass;
the KV-cache edit modifies the stored context from prior tokens), their combined
effect at token 0 should be additive (each contributes its full effect with
no interaction). For short generations (< 50 tokens), the KV-cache contamination
from residual steering has not yet accumulated, so both effects are preserved
at >= 85% of solo. For long generations (> 200 tokens), the residual edit has
repeatedly contaminated the KV cache, and the KV-cache steering's contextual
frame is disrupted. Formal claim: at <= 50 generated tokens, joint success
rate >= 85% for both residual and KV-cache effects; at >= 200 generated tokens,
joint success drops below 80% for the KV-cache effect specifically (contamination-
driven degradation), at 3-seed median on Gemma-2-2B-it.

---

## 3. Falsifier (>= 30 words)

If joint success at <= 50 tokens is below 85% for either method (early-phase
composition is already disrupted), the disjoint-site stacking prediction is
FALSIFIED. If joint success at >= 200 tokens does not drop below 80% for the
KV-cache method (no contamination-driven degradation at long generation), the
KV-contamination mechanism is not active and the length-dependent degradation
prediction is FALSIFIED.

---

## 4. Citations (Citation Rigor >= 80 words)

```
(anon) 2025 arXiv 'Key-Orthogonal Projection Steering' (SKOP, arXiv:2605.06342)
— introduced to prevent residual-stream steering from contaminating the KV
cache; the existence of SKOP is evidence that contamination is a real phenomenon;
this experiment tests the unmitigated contamination and the SKOP mitigation.

(anon) 2025 arXiv 'KV-Cache Steering for Language Model Control'
(arXiv:2507.08799) [NEEDS VERIFICATION] — the KV-cache steering method being
combined with residual CAA; provides the disjoint-site composition target.

[Steering-stackable-vs-competing-analysis.md §3.4, this project]: "Residual-
stream steering can contaminate the KV cache for downstream tokens. SKOP and
GCAD are explicit fixes — meaning unmitigated residual steering competes with
clean attention routing over long generations."

[N5 geometry result, C2, this project, SUPPORTED]: logPPL = 5.40 + 2.87 *
offshell, R² = 0.81 — KV-cache contamination is a secondary offshell mechanism:
the contaminated keys/values pull future hidden states off-manifold; the N5
law should predict cumulative PPL degradation over generation length.
```

---

## 5. Mechanism

### 5.1 Contamination mechanism

At generation step t with residual steering:

    h_t = model(h_{t-1}) + alpha * v   (steered residual)
    k_t = W_K * h_t                    (key from steered h_t)
    v_t = W_V * h_t                    (value from steered h_t)

The keys k_t and values v_t now carry the steering edit. At step t+1, the
attention mechanism reads the KV cache including k_t and v_t:

    Attn_{t+1} = softmax(Q_{t+1} K_{1:t}^T / sqrt(d)) * V_{1:t}

The steering-contaminated k_t and v_t influence all future attention computations.
Over T tokens, the contamination accumulates.

### 5.2 SKOP mitigation

SKOP projects the residual edit out of the key direction before it reaches
the key computation:

    h_t_safe = h_t - (h_t · k_mean) * k_mean / ||k_mean||^2
    k_t_safe = W_K * h_t_safe  (contamination-free key)

This prevents the edit from leaking into the KV cache while preserving it in
the residual stream for behavior purposes.

---

## 6. Predicted Delta

| Metric | Predicted value | Rationale |
|---|---|---|
| Joint success (residual, <= 50 tokens) | >= 85% of solo | Disjoint sites, short generation |
| Joint success (KV-cache, <= 50 tokens) | >= 85% of solo | Disjoint sites, short generation |
| Joint success (KV-cache, >= 200 tokens) | < 80% of solo | KV contamination accumulates |
| Joint success (KV-cache, >= 200 tokens, +SKOP) | >= 85% of solo | SKOP mitigates contamination |
| PPL degradation vs generation length (unmitigated) | Increasing (monotone) | N5 cumulative contamination |

Pre-registered; [NEEDS VERIFICATION] on Gemma-2-2B.

---

## 7. Experimental Protocol

### 7.1 Primary experiment

- **Model:** Gemma-2-2B-it, 4-bit; smoke on Gemma-3-1B-it
- **Conditions:** (1) residual solo, (2) KV-cache solo, (3) joint unmitigated,
  (4) joint + SKOP
- **Generation lengths:** {20, 50, 100, 200, 400} tokens per condition
- **Metrics:** behavior success rate per condition per length; PPL per length;
  KV-cache contamination (measured as cross-attention to steered vs clean keys)
- **Seeds:** 3

### 7.2 Where it shines

The length-dependent degradation is most visible at long generation (400 tokens)
and high alpha (maximizing contamination). Short-generation baseline is important
to establish the initial composition quality.

---

## 8. Cross-References

- **E25** (DoLa stacking): DoLa operates at the logit level (post-KV-cache);
  complements residual + KV-cache without contaminating either
- **N5** (norm budget, SUPPORTED): cumulative KV contamination is a secondary
  off-shell mechanism
- **E17** (near-orthogonal stacking): residual + KV-cache is the disjoint-site
  version of E17's same-site orthogonal stacking
- **IDEA_TABLE.md** Block C row E24

---

## 9. Committee Q&A

**Q: Is the KV-cache steering method (arXiv:2507.08799) verified?**

> Marked [NEEDS VERIFICATION]. If this paper is unverified, we can implement
> a simplified KV-cache edit (directly modifying cached key/value tensors via
> hooks) as a stand-in for the full method. The contamination experiment does
> not require the full KV-cache steering paper to be valid — it only requires
> that we can edit cached KV tensors, which is a standard HuggingFace hook.

---

## 10. Verification Checklist

- [ ] KV-cache edit hook implemented (verified on toy example)
- [ ] SKOP key-orthogonal projection implemented
- [ ] Joint success measured at 5 generation lengths for all conditions
- [ ] PPL per length plotted; monotone increase confirmed for unmitigated condition
- [ ] SKOP mitigated condition compared to unmitigated at >= 200 tokens
- [ ] IDEA_TABLE.md row E24 updated

---

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: UNTESTED / PENDING. N5 (SUPPORTED)
  predicts cumulative PPL growth from contamination. arXiv:2507.08799 marked
  [NEEDS VERIFICATION]. SKOP (arXiv:2605.06342) is the verified mitigation.
  Code needed: KV-cache edit hook; SKOP projection; generation-length sweep.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-C.*

### Prior plausibility

HIGH for the short-generation disjoint-site composition (algebraically clean).
MEDIUM for the length-dependent degradation (depends on how much the KV-cache
contamination actually influences attention, which varies by architecture and
generation regime).

### Confounds

1. **Attention pattern overlap:** if the model attends primarily to the last
   few tokens (recent bias), KV-cache contamination from early tokens may not
   propagate to late token attention.
2. **SKOP calibration:** the key-orthogonal direction must be estimated; an
   imprecise estimate will leave residual contamination.

### Verdict

**NOVEL+TESTABLE.** The generation-length sweep is the key experimental design.
The SKOP mitigation provides a clean remediation arm that makes the experiment
more than just a documentation of failure.

---

## Pseudocode & Methodology

This section specializes [`../METHODOLOGY.md`](../METHODOLOGY.md) to disjoint-site
composition (residual CAA + KV-cache steering) with a generation-length sweep and the
SKOP mitigation. Status: `UNTESTED` — needs a KV-cache edit hook + SKOP projection.

### 1. Steering-vector recipe (one residual DiffMean + a KV-cache edit)

```python
# Residual arm: standard closed-form DiffMean (METHODOLOGY §1.3), injected at layer L.
v = normalize(extract.build_vector_bank(model,tok,load_concept(behavior),L)[L]["diffmean"])
# KV-cache arm: a disjoint-site edit on the cached key/value tensors (separate hook).
# SKOP mitigation (§5.2): project the residual edit out of the key direction before W_K:
#   h_safe = h - (h · k_mean) * k_mean / ||k_mean||^2   ->  contamination-free key
```

### 2. Experiment procedure

```text
conditions = { residual_solo  : hooks.apply_operation(h, v, "add", alpha),
               kvcache_solo    : edit cached (k,v) tensors,
               joint_unmitigated: both,
               joint_SKOP      : both, with key-orthogonal projection on residual h }
for length in {20, 50, 100, 200, 400}:        # the discriminative axis
  for cond in conditions:
    for seed in 1..3:
      generate(length tokens); measure behavior_success per method, PPL,
      geometry.offshell_displacement   # KV contamination is a SECONDARY offshell source (N5)
```

### 3. Measurement & decision rule

- PRIMARY metric: joint behavior success (per method) vs solo, as a function of
  generation length.
- Pre-registered FALSIFIER (§3): at `<=50` tokens joint success `< 85%` for either
  method ⇒ disjoint-site stacking FALSIFIED; at `>=200` tokens KV-cache success does
  NOT drop below 80% ⇒ contamination mechanism FALSIFIED.
- Remediation arm: joint+SKOP restores `>=85%` at `>=200` tokens (§6).

### 4. Where the code is / status

The residual `add` path exists. MISSING: a **KV-cache edit hook** (standard HF cache
hook — a simplified direct (k,v)-tensor edit can stand in for the unverified
arXiv:2507.08799 method), the **SKOP key-orthogonal projection**, and the
generation-length sweep harness. `UNTESTED`.

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/E24.md`.
